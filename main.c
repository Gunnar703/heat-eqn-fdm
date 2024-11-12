#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Stores information about the problem being solved
typedef struct {
    size_t num_pts;
    size_t max_timesteps;
    double diffusion_number;
    double dirichlet_bc_left;
    double neumann_bc_right;
    double thickness;
    double density;
    double spec_heat;
    double conductivity;
    double initial_cond;
    double stopping_temp;
} ProblemDefinition;

// 2D matrix stored contiguously in memory
typedef struct {
    size_t nrows;
    size_t ncols;
    double *data;
} DoubleMatrix;

int print_usage(void);  // prints help in case program is executed wrong  
int parse_args(int argc, char** argv, ProblemDefinition* definition);  // parses CLI arguments
void print_problem_def(ProblemDefinition *definition);  // prints the parsed problem definition

DoubleMatrix alloc_matrix(size_t nrows, size_t ncols);
static inline double matrix_get(DoubleMatrix* matrix, size_t i, size_t j);
static inline int matrix_set(DoubleMatrix* matrix, size_t i, size_t j, double x);
static inline int matrix_free(DoubleMatrix* matrix);
int solve_tridiagonal(DoubleMatrix* a, DoubleMatrix* b, DoubleMatrix* c, DoubleMatrix* d, DoubleMatrix* d_prime, DoubleMatrix* c_prime, DoubleMatrix* x);

int matrix_write(DoubleMatrix* matrix, char* file_name, char delimiter);
int vector_append(DoubleMatrix* matrix, FILE* fptr, char delimiter);

// *********** //
// ENTRY POINT //
// *********** //
int main(int argc, char** argv) {
    int result = 0;
    size_t i, j;

    // Extract arguments
    ProblemDefinition definition;
    result = parse_args(argc, argv, &definition);
    if (result != 0) return result;
    print_problem_def(&definition);

    // Compute spacing from input args
    double diffusivity = definition.conductivity 
                       / definition.density
                       / definition.spec_heat;
    double dx = definition.thickness / ( definition.num_pts - 1 );
    double dt = definition.diffusion_number * pow(dx, 2) / diffusivity;

    // Allocate "scratchpad" arrays
    DoubleMatrix a       = alloc_matrix(definition.num_pts, 1);
    DoubleMatrix b       = alloc_matrix(definition.num_pts, 1);
    DoubleMatrix c       = alloc_matrix(definition.num_pts, 1);
    DoubleMatrix d       = alloc_matrix(definition.num_pts, 1);
    DoubleMatrix x       = alloc_matrix(definition.num_pts, 1);
    DoubleMatrix c_prime = alloc_matrix(definition.num_pts, 1);
    DoubleMatrix d_prime = alloc_matrix(definition.num_pts, 1);

    // Open solution files
    FILE* soln_fptr = fopen("solution.dat", "w");
    FILE* time_fptr = fopen("time.dat", "w");
    
    if (soln_fptr == NULL) { result = -1; goto defer; }
    if (time_fptr == NULL) { result = -1; goto defer; }

    for (j = 0; j < x.nrows; j++) {
        result = matrix_set(&x, j, 0, definition.initial_cond);
        if (result != 0) goto defer;
    }

    result = vector_append(&x, soln_fptr, ' '); if (result != 0) goto defer;
    fprintf(time_fptr, "%.5lf\n", 0 * dt);

    // Set up a, b, and d since they stay constant
    const size_t N  = d.nrows - 1;
    const double dd = 1 + 2 * definition.diffusion_number;
    const double bb = -definition.diffusion_number;
    for (j = 0; j < d.nrows; j++) {
        // Set d
        result = matrix_set(&d, j, 0, dd); if (result != 0) goto defer;

        // Set b
        if (j >= 1) {
            result = matrix_set(&b, j, 0, bb); if (result != 0) goto defer;
        }

        // Set a
        if (j < N) {
            result = matrix_set(&a, j, 0, bb); if (result != 0) goto defer;
        }
    }

    // Overwrite endpoints to enforce BCs
    result = matrix_set(&d, 0, 0, 1.0); if (result != 0) goto defer;
    result = matrix_set(&a, 0, 0, 0.0); if (result != 0) goto defer;

    const double b_end = 1 - 2 * definition.diffusion_number;
    const double d_end = 2 * definition.diffusion_number;
    result = matrix_set(&b, N, 0, b_end); if (result != 0) goto defer;
    result = matrix_set(&d, N, 0, d_end); if (result != 0) goto defer;

    for (i = 1; i < definition.max_timesteps; i++) {
        if (((i + 1) % 100 == 0) || (i == 1) || (i + 1 == definition.max_timesteps) ) {
            printf("[INFO] Timestep Number %d / %d\n", i + 1, definition.max_timesteps);
        }

        // Set current solution to {x}
        for (j = 1; j < c.nrows - 1; j++) {
            result = matrix_set(&c, j, 0, matrix_get(&x, j, 0));
        }

        const double left_end_temp_at_tmstp_start = matrix_get(&x, x.nrows - 1, 0);

        // Write endpoints to enforce BCs
        const double dirichlet_bc = definition.dirichlet_bc_left;
        const double neumann_bc   = definition.neumann_bc_right
                                    * definition.diffusion_number
                                    * 2 * dx
                                    + matrix_get(&c, c.nrows - 2, 0);

        result = matrix_set(&c, 0, 0, dirichlet_bc); if (result != 0) goto defer;
        result = matrix_set(&c, c.nrows - 1, 0, neumann_bc); if (result != 0) goto defer;

        // Solve system [A]{x} = {c} where 
        // [A] = [d0 a0 0  0   0  0 ]
        //       [b1 d1 a1 0   0  0 ]
        //       [ 0 b2 d2 a2  0  0 ]
        //       [ 0  0  *  *  *  0 ]
        //       [ 0  0  0  bN dN aN]
        result = solve_tridiagonal(&a, &b, &c, &d, &c_prime, &d_prime, &x); if (result != 0) goto defer;

        // Append current solution and time steps to files
        result = vector_append(&x, soln_fptr, ' '); if (result != 0) goto defer;
        fprintf(time_fptr, "%.5lf\n", dt * i);

        // If stopping after a left-side temperature is enabled, check for it here
        const double left_end_temp_at_tmstp_end = matrix_get(&x, x.nrows - 1, 0);

        if ((definition.stopping_temp > 0) && (left_end_temp_at_tmstp_end > definition.stopping_temp)) {
            const double prev_time = (i - 1) * dt;
            const double crossing_time = prev_time 
                                        + (definition.stopping_temp - left_end_temp_at_tmstp_start)
                                        / (left_end_temp_at_tmstp_end - left_end_temp_at_tmstp_start)
                                        * dt;
            printf("[INFO] Crossing Time: %.10g\n", crossing_time);
            break;
        }

    }

defer:
    if (result != 0) printf("ERROR: Program exited with code %d.\n", result);
    matrix_free(&a);
    matrix_free(&b);
    matrix_free(&c);
    matrix_free(&d);
    matrix_free(&x);
    matrix_free(&c_prime);
    matrix_free(&d_prime);
    fclose(soln_fptr);
    fclose(time_fptr);
    return result;
}

int print_usage(void) {
    printf("USAGE: implicitHeatEqn <1> <2> <3> <4> <5> <6> <7> <8> <9> <10>\n");
    printf("    1     number of points in the discretization\n");
    printf("    2     maximum number of time steps\n");
    printf("    3     diffusion number\n");
    printf("    4     left side dirichlet boundary condition\n");
    printf("    5     right side neumann boundary condition\n");
    printf("    6     thickness of the plate\n");
    printf("    7     density of the plate\n");
    printf("    8     specific heat\n");
    printf("    9     thermal conductivity\n");
    printf("    10    initial temperature\n");
    printf("    11    left-side temperature at which simulation will be stopped. -1 indicates run to end.\n");
    return 0;
}

int parse_args(int argc, char** argv, ProblemDefinition* definition) {
    if (argc != 12) {
        print_usage();
        return -1;
    }

    definition->num_pts           = atoi(argv[1]);
    definition->max_timesteps     = atoi(argv[2]);
    definition->diffusion_number  = atof(argv[3]);
    definition->dirichlet_bc_left = atof(argv[4]);
    definition->neumann_bc_right  = atof(argv[5]);
    definition->thickness         = atof(argv[6]);
    definition->density           = atof(argv[7]);
    definition->spec_heat         = atof(argv[8]);
    definition->conductivity      = atof(argv[9]);
    definition->initial_cond      = atof(argv[10]);
    definition->stopping_temp     = atof(argv[11]);
    return 0;
}

void print_problem_def(ProblemDefinition *definition) {
    printf("---------- PARAMETERS ---------\n");
    printf("      NUM PTS    %d\n", definition->num_pts);
    printf("NUM TIMESTEPS    %d\n", definition->max_timesteps);
    printf("  DIFF NUMBER    %.4g\n", definition->diffusion_number);
    printf("LEFT END TEMP    %.4g\n", definition->dirichlet_bc_left);
    printf("RIGHT END FLX    %.4g\n", definition->neumann_bc_right);
    printf("    THICKNESS    %.4g\n", definition->thickness);
    printf("      DENSITY    %.4g\n", definition->density);
    printf("SPECIFIC HEAT    %.4g\n", definition->spec_heat);
    printf(" CONDUCTIVITY    %.4g\n", definition->conductivity);
    printf(" INITIAL TEMP    %.4g\n", definition->initial_cond);
    printf("STOPPING TEMP    %.4g\n", definition->stopping_temp);
    printf("-------- END PARAMETERS -------\n");
}

// Allocates a matrix of doubles
// Returns NULL if allocation fails
DoubleMatrix alloc_matrix(size_t nrows, size_t ncols) {
    DoubleMatrix matrix;
    matrix.nrows = nrows;
    matrix.ncols = ncols;
    matrix.data = (double*)calloc(nrows*ncols, sizeof(double));
    return matrix;
}

// Frees a matrix
static inline int matrix_free(DoubleMatrix* matrix) {
    free(matrix->data);
    return 0;
}

// Indexes into the provided matrix
static inline double matrix_get(DoubleMatrix* matrix, size_t i, size_t j) {
    if (i > matrix->nrows - 1) goto defer;
    if (j > matrix->ncols - 1) goto defer;

    size_t lin_idx = i * matrix->ncols + j;
    return matrix->data[lin_idx];

defer:
    printf("[ERROR] Provided indices are out of bounds. Returning -1.");
    return -1.0;
}

// Sets element i, j of the provided matrix
static inline int matrix_set(DoubleMatrix* matrix, size_t i, size_t j, double x) {
    if (i > matrix->nrows - 1) return -1;
    if (j > matrix->ncols - 1) return -1;

    size_t lin_idx = i * matrix->ncols + j;
    matrix->data[lin_idx] = x;

    return 0;
}

// Writes matrix to file
int matrix_write(DoubleMatrix* matrix, char* file_name, char delimiter) {
    int result = 0;
    size_t i, j;
    
    // Open the file
    FILE* fptr = fopen(file_name, "w");
    if (fptr == NULL) { result = -1; goto defer; }
    
    for (i = 0; i < matrix->nrows; i++) {
        for (j = 0; j < matrix->ncols; j++) {
            fprintf(fptr, "%.5lf%c", matrix_get(matrix, i, j), delimiter);
        }
        fprintf(fptr, "\n");
    }

defer:
    fclose(fptr);
    return result;
}

// Appends a vector (N x 1 or 1 x N matrix) to file as a row vector
int vector_append(DoubleMatrix* matrix, FILE* fptr, char delimiter) {
    if (fptr == NULL) return -1;
    if ((matrix->nrows != 1) && (matrix->ncols != 1)) return -1;

    size_t i;
    if (matrix->nrows == 1) {
        for (i = 0; i < matrix->ncols; i++) 
            fprintf(fptr, "%.5lf%c", matrix_get(matrix, 0, i), delimiter);
    } else {
        for (i = 0; i < matrix->nrows; i++) 
            fprintf(fptr, "%.5lf%c", matrix_get(matrix, i, 0), delimiter);
    }

    fprintf(fptr, "\n");

    return 0;
}

// Solve system [A]{x} = {c} where 
// ```
// [A] = [d0 a0 0  0   0  0 ]
//       [b1 d1 a1 0   0  0 ]
//       [ 0 b2 d2 a2  0  0 ]
//       [ 0  0  *  *  *  0 ]
//       [ 0  0  0  bN dN aN]
// ```
// in pre-allocated scratch space (d_prime, c_prime) using Thomas Algorithm
// Returns solution in last argument, {x}
int solve_tridiagonal(DoubleMatrix* a, DoubleMatrix* b, DoubleMatrix* c, DoubleMatrix* d, DoubleMatrix* c_prime, DoubleMatrix* d_prime, DoubleMatrix* x) {
    int result = 0;
    size_t i;
    size_t N = d->nrows - 1;

    // Initialize d'(0) and c'(0)
    result = matrix_set(d_prime, 0, 0, matrix_get(d, 0, 0)); if (result != 0) return result;
    result = matrix_set(c_prime, 0, 0, matrix_get(c, 0, 0)); if (result != 0) return result;

    // Forward sweep
    for (i = 1; i < d->nrows; i++) {
        const double m  = matrix_get(b, i, 0) / matrix_get(d_prime, i - 1, 0);
        const double dp = matrix_get(d, i, 0) - m * matrix_get(a, i - 1, 0);
        const double cp = matrix_get(c, i, 0) - m * matrix_get(c_prime, i - 1, 0);

        result = matrix_set(d_prime, i, 0, dp); if (result != 0) return result;
        result = matrix_set(c_prime, i, 0, cp); if (result != 0) return result;
    }

    // Backward Sweep
    const double last_x = matrix_get(c_prime, N, 0) / matrix_get(d_prime, N, 0);
    result = matrix_set(x, N, 0, last_x);
    for (i = 1; i <= N; i++) {
        const double xx = (
            matrix_get(c_prime, N - i, 0) 
            - matrix_get(a, N - i, 0) 
            * matrix_get(x, N - i + 1, 0)
        ) / matrix_get(d_prime, N - i, 0);
        result = matrix_set(x, N - i, 0, xx); if (result != 0) return result;
    }

    return result;
}