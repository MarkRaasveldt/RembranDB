

#include <llvm-c/Core.h>
#include <llvm-c/ExecutionEngine.h>
#include <llvm-c/Target.h>
#include <llvm-c/Analysis.h>
#include <llvm-c/BitWriter.h>
#include <llvm-c/Transforms/IPO.h>
#include <llvm-c/Transforms/Scalar.h>
#include <llvm-c/Transforms/Vectorize.h>

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include <ctype.h>

#include "table.h"
#include "parser.h"

#include "target_machine.h"

static void Initialize(void);
static char* ReadQuery(void);
static Table *ExecuteQuery(Query *query);
static void Cleanup(void); 

static bool enable_optimizations = false;
static bool print_result = true;
static bool print_llvm = true;
static bool execute_statement = false;
static char* statement;

#define OVERFLOW_CODE 1

// long long loop(double* result, double **inputs, long long size)
// returns: number of elements inserted
typedef lng (*fptr)(double*,double**,lng);

static LLVMPassManagerRef InitializePassManager(LLVMModuleRef module);

static LLVMValueRef
PerformOperation(Operation *op, LLVMBuilderRef builder, LLVMValueRef index) {
    if (op->type == OPTYPE_binop) {
        BinaryOperation *binop = (BinaryOperation*) op;
        LLVMValueRef left = PerformOperation(binop->left, builder, index);
        LLVMValueRef right = PerformOperation(binop->right, builder, index);
        switch(binop->optype) {
            case OPTYPE_mul: return LLVMBuildFMul(builder, left, right, "x*y");
            case OPTYPE_div: return LLVMBuildFDiv(builder, left, right, "x/y");
            case OPTYPE_add: return LLVMBuildFAdd(builder, left, right, "x+y");
            case OPTYPE_sub: return LLVMBuildFSub(builder, left, right, "x-y");
            case OPTYPE_lt:  return LLVMBuildFCmp(builder, LLVMRealOLT, left, right, "x<y");
            case OPTYPE_le:  return LLVMBuildFCmp(builder, LLVMRealOLE, left, right, "x<y");
            case OPTYPE_eq:  return LLVMBuildFCmp(builder, LLVMRealOEQ, left, right, "x<y");
            case OPTYPE_ne:  return LLVMBuildFCmp(builder, LLVMRealONE, left, right, "x<y");
            case OPTYPE_gt:  return LLVMBuildFCmp(builder, LLVMRealOGT, left, right, "x<y");
            case OPTYPE_ge:  return LLVMBuildFCmp(builder, LLVMRealOGE, left, right, "x<y");
            case OPTYPE_and: return LLVMBuildAnd(builder, left, right, "x && y");
            case OPTYPE_or:  return LLVMBuildOr(builder, left, right, "x || y");
        }
    } else if (op->type == OPTYPE_colmn) {
        LLVMValueRef colptr_base = LLVMBuildLoad(builder, ((ColumnOperation*)op)->column->llvm_ptr, "&col");
        LLVMValueRef colptr_offset = LLVMBuildGEP(builder, colptr_base, &index, 1, "&col[index]");
        return LLVMBuildLoad(builder, colptr_offset, "col[index]");
    } else if (op->type == OPTYPE_const) {
        return LLVMConstReal(LLVMDoubleType(), ((ConstantOperation*)op)->value);
    }
    return NULL;
}

static Table*
ExecuteQuery(Query *query) {
    size_t i;

    clock_t tic = clock();
    LLVMModuleRef module = LLVMModuleCreateWithName("LoopModule");
    LLVMOptimizeModuleForTarget(module);

    LLVMTypeRef double_type = LLVMDoubleType();
    LLVMTypeRef doubleptr_type = LLVMPointerType(double_type, 0);
    LLVMTypeRef doubleptrptr_type = LLVMPointerType(doubleptr_type, 0);
    LLVMTypeRef int64_type = LLVMInt64Type();

    LLVMTypeRef return_type = int64_type;
    LLVMTypeRef param_types[] = { doubleptr_type, doubleptrptr_type, 
        int64_type};

    LLVMTypeRef prototype = LLVMFunctionType(return_type, param_types, 3, 0);
    LLVMValueRef function = LLVMAddFunction(module, "loop", prototype);

    LLVMBuilderRef builder = LLVMCreateBuilder();

    LLVMBasicBlockRef entry = LLVMAppendBasicBlock(function, "entry");
    LLVMBasicBlockRef condition = LLVMAppendBasicBlock(function, "condition");
    LLVMBasicBlockRef body_condition = NULL;
    if (query->where) {
        body_condition = LLVMAppendBasicBlock(function, "body_condition");
    }
    LLVMBasicBlockRef body_main = LLVMAppendBasicBlock(function, "body_main");
    LLVMBasicBlockRef body_store = LLVMAppendBasicBlock(function, "body_store");
    LLVMBasicBlockRef increment = LLVMAppendBasicBlock(function, "increment");
    LLVMBasicBlockRef end = LLVMAppendBasicBlock(function, "end");
    LLVMBasicBlockRef overflow_error = LLVMAppendBasicBlock(function, "overflow_error");

    size_t columns = GetColCount(query->columns);
    size_t elements = query->columns->column->size;

    LLVMValueRef index_addr;
    LLVMValueRef result_index_addr = NULL;
    LLVMPositionBuilderAtEnd(builder, entry);
    {
        if (query->where) {
            // if there is a WHERE condition we also need to keep track of the amount of elements
            result_index_addr = LLVMBuildAlloca(builder, int64_type, "result_index");
            LLVMBuildStore(builder, LLVMConstInt(int64_type, 0, 1), result_index_addr);
        }
        i = 0;
        for(ColumnList *col = query->columns; col; col = col->next, i++) {
            if (col->column) {
                LLVMValueRef constone = LLVMConstInt(int64_type, i, true);
                LLVMValueRef colptrptr = LLVMBuildGEP(builder, LLVMGetParam(function, 1), &constone, 1, "&inputs[i]");
                LLVMValueRef colptr = LLVMBuildLoad(builder, colptrptr, "inputs[i]");
                col->column->llvm_ptr = LLVMBuildAlloca(builder, doubleptr_type, "col*");
                LLVMBuildStore(builder, colptr, col->column->llvm_ptr);

            }
        }
        index_addr = LLVMBuildAlloca(builder, int64_type, "index");
        LLVMBuildStore(builder, LLVMConstInt(int64_type, 0, 1), index_addr);
        LLVMBuildBr(builder, condition);
    }

    LLVMPositionBuilderAtEnd(builder, condition);
    {
        LLVMValueRef index = LLVMBuildLoad(builder, index_addr, "[index]");
        LLVMValueRef cond = LLVMBuildICmp(builder, LLVMIntSLT, index, LLVMGetParam(function, 2), "index < size");
        LLVMBuildCondBr(builder, cond, query->where ? body_condition : body_main, end);
    }
    if (query->where) {
        LLVMPositionBuilderAtEnd(builder, body_condition);
        {
            // this is basically the WHERE clause
            // if the WHERE clause is true we compute the SELECT
            LLVMValueRef index = LLVMBuildLoad(builder, index_addr, "[index]");

            // todo: operation
            LLVMValueRef where_condition = PerformOperation(query->where, builder, index);

            LLVMBuildCondBr(builder, where_condition, body_main, increment);
        }
    }

    LLVMValueRef index_body;
    LLVMValueRef result_value;
    LLVMPositionBuilderAtEnd(builder, body_main);
    {   
        // this computes the SELECT clause
        index_body = LLVMBuildLoad(builder, index_addr, "[index]");

        // todo: operation
        result_value = PerformOperation(query->select, builder, index_body);

        LLVMValueRef overflow_occurred = LLVMBuildFCmp(builder, LLVMRealOEQ, result_value, LLVMConstReal(double_type, INFINITY), "cmp");
        LLVMBuildCondBr(builder, overflow_occurred, overflow_error, body_store);
    }

    LLVMPositionBuilderAtEnd(builder, body_store);
    {   
        LLVMValueRef result_index = index_body;
        if (query->where) {
            // if we have a WHERE clause the current result index can differ from the index
            result_index = LLVMBuildLoad(builder, result_index_addr, "[result_index]");
        } 
        LLVMValueRef result_addr = LLVMBuildGEP(builder, LLVMGetParam(function, 0), &result_index, 1, "&result[result_index]");
        LLVMBuildStore(builder, result_value, result_addr);
        if (query->where) {
            // increment result index
            LLVMValueRef result_indexpp = LLVMBuildAdd(builder, result_index, LLVMConstInt(int64_type, 1, 1), "index++");
            LLVMBuildStore(builder, result_indexpp, result_index_addr);
        }
        LLVMBuildBr(builder, increment);
    }

    LLVMPositionBuilderAtEnd(builder, overflow_error);
    {
        LLVMBuildRet(builder, LLVMConstInt(int64_type, OVERFLOW_CODE, 1));
    }
    LLVMPositionBuilderAtEnd(builder, increment);
    {
        LLVMValueRef index = LLVMBuildLoad(builder, index_addr, "[index]");
        LLVMValueRef indexpp = LLVMBuildAdd(builder, index, LLVMConstInt(int64_type, 1, 1), "index++");
        LLVMBuildStore(builder, indexpp, index_addr);
        LLVMBuildBr(builder, condition);
    }
    LLVMPositionBuilderAtEnd(builder, end);
    {
        LLVMValueRef elements = NULL;
        if (query->where) {
            elements = LLVMBuildLoad(builder, result_index_addr, "[result_index]");
        } else {
            elements = LLVMBuildLoad(builder, index_addr, "[index]");
        }

        LLVMBuildRet(builder, elements);
    }

    LLVMPassManagerRef passManager = InitializePassManager(module);
    LLVMRunFunctionPassManager(passManager, function);
    
    LLVMDumpModule(module);

    struct LLVMMCJITCompilerOptions options;
    LLVMInitializeMCJITCompilerOptions(&options, sizeof(options));
    LLVMExecutionEngineRef engine;
    char *error = NULL;
    if (LLVMCreateMCJITCompilerForModule(&engine, module, &options, sizeof(options), &error) != 0) {
        fprintf(stderr, "failed to create execution engine\n");
        abort();
    }
    if (error) {
        fprintf(stderr, "error: %s\n", error);
        LLVMDisposeMessage(error);
        exit(EXIT_FAILURE);
    }

    fptr loop_func = (fptr) LLVMGetFunctionAddress(engine, "loop");
    if (!loop_func) {
        printf("Failed to get function pointer.\n");
        exit(1);
    }
    clock_t toc = clock();
    printf("Compilation: %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);
    double **inputs = malloc(sizeof(double*) * columns);
    i = 0;
    for(ColumnList *col = query->columns; col; col = col->next, i++) {
        if (col->column) {
            inputs[i] = col->column->data;
        }
    }
    double *result = malloc(sizeof(double) * elements);
    size_t res_elements = loop_func(result, inputs, elements);
    if (res_elements == OVERFLOW_CODE) {
        fprintf(stderr, "ERROR: Overflow in calculation!\n");
        return NULL;
    }

    Column *column = CreateColumn(result, res_elements);
    return CreateTable("Result", column);
}

int main(int argc, char** argv) {
    for(int i = 1; i < argc; i++) {
        char *arg = argv[i];
        if (strcmp(arg, "--help") == 0) {
            printf("RembranDB Options.\n");
            printf("  -opt              Enable  LLVM optimizations.\n");
            printf("  -no-print         Do not print query results.\n");
            printf("  -no-llvm          Do not print LLVM instructions.\n");
            printf("  -s \"stmnt\"        Execute \"stmnt\" and exit.\n");
            return 0;
        } else if (strcmp(arg, "-opt") == 0) {
            printf("Optimizations enabled.\n");
            enable_optimizations = true;
        } else if (strcmp(arg, "-no-print") == 0) {
            printf("Printing output disabled.\n");
            print_result = false;
        } else if (strcmp(arg, "-no-llvm") == 0) {
            printf("Printing LLVM disabled.\n");
            print_llvm = false;
        } else if (strcmp(arg, "-s") == 0) {
            execute_statement = true;
        } else if (execute_statement) {
            statement = arg;
        } else {
            printf("Unrecognized command line option \"%s\".\n", arg);
            exit(1);
        }
    }
    if (!execute_statement) {
        printf("# RembranDB server v0.0.0.1\n");
        printf("# Serving table \"demo\", with no support for multithreading\n");
        printf("# Did not find any available memory (didn't look for any either)\n");
        printf("# Not listening to any connection requests.\n");
        printf("# RembranDB/SQL module loaded\n");
    }
    Initialize();

    while(true) {
        char *query_string;
        if (!execute_statement) {
            query_string = ReadQuery();
        } else {
            query_string = statement;
        }

        if (strcmp(query_string, "\\q") == 0 || (strlen(query_string) > 0 && query_string[0] == '^')) break;
        if (strcmp(query_string, "\\d") == 0) {
            PrintTables();
            continue;
        }
        Query *query = ParseQuery(query_string);
        
        if (query) {
            clock_t tic = clock();
            Table *tbl = ExecuteQuery(query);
            clock_t toc = clock();

            printf("Total Runtime: %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);

            if (print_result) {
                PrintTable(tbl);
            }
        }
        if (execute_statement) break;
    }

    Cleanup();
}

static LLVMPassManagerRef InitializePassManager(LLVMModuleRef module) {
    LLVMPassManagerRef passManager = LLVMCreateFunctionPassManagerForModule(module);
    // This set of passes was copied from the Julia people (who probably know what they're doing)
    // Julia Passes: https://github.com/JuliaLang/julia/blob/master/src/jitlayers.cpp

    LLVMAddTargetMachinePasses(passManager);
    LLVMAddCFGSimplificationPass(passManager);
    LLVMAddPromoteMemoryToRegisterPass(passManager);
    LLVMAddInstructionCombiningPass(passManager);
    LLVMAddScalarReplAggregatesPass(passManager);
    LLVMAddScalarReplAggregatesPassSSA(passManager);
    LLVMAddInstructionCombiningPass(passManager);
    LLVMAddJumpThreadingPass(passManager);
    LLVMAddInstructionCombiningPass(passManager);
    LLVMAddReassociatePass(passManager);
    LLVMAddEarlyCSEPass(passManager);
    LLVMAddLoopIdiomPass(passManager);
    LLVMAddLoopRotatePass(passManager);
    LLVMAddLICMPass(passManager);
    LLVMAddLoopUnswitchPass(passManager);
    LLVMAddInstructionCombiningPass(passManager);
    LLVMAddIndVarSimplifyPass(passManager);
    LLVMAddLoopDeletionPass(passManager);
    LLVMAddLoopUnrollPass(passManager);
    LLVMAddLoopVectorizePass(passManager);
    LLVMAddInstructionCombiningPass(passManager);
    LLVMAddGVNPass(passManager);
    LLVMAddMemCpyOptPass(passManager);
    LLVMAddSCCPPass(passManager);
    LLVMAddInstructionCombiningPass(passManager);
    LLVMAddSLPVectorizePass(passManager);
    LLVMAddAggressiveDCEPass(passManager);
    LLVMAddInstructionCombiningPass(passManager);

    LLVMInitializeFunctionPassManager(passManager);
    return passManager;
}

static void Initialize(void) {
    // LLVM initialization code
    LLVMLinkInMCJIT();
    LLVMInitializeNativeTarget();
    LLVMInitializeAllTargetMCs();
    LLVMInitializeAllAsmPrinters();
    LLVMInitializeAllAsmParsers();
    // Load data, demo table = small table (20 entries per column)
    InitializeTable("demo");
}

static char *
ReadQuery(void) {
    char *buffer = malloc(5000 * sizeof(char));
    size_t buffer_pos = 0;
    char c;
    printf("> ");
    while((c = getchar()) != EOF) {
        if (c == '\n') {
            if (buffer[0] == '\\') {
                return buffer;
            } else {
                buffer[buffer_pos++] = ' ';
                printf("> ");
                continue;
            }
        } else if (c == ';') {
            buffer[buffer_pos++] = '\0';
            return buffer;
        } else {
            buffer[buffer_pos++] = c;
        }
    }
    return strdup("\\q");
}

static void 
Cleanup(void) {

}
