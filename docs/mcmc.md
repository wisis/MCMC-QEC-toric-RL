# MCMC
    This class implements the MCMC sampling algorithm for probability based reward scheme.
    Defines rule_table

## apply_stabilizer(toric_model, row=int, col=int, operator=int)
    A function that saves stabilizers on toric_model

## test_apply_stabilizer(qubit_matrix, size=int, row=int, col=int, operator=int)
    A function that applies stabilizer on qubit_matrix

## test_apply_random_stabilizer(qubit_matrix, size)
    A function that runs apply_random_stabilizer
    
## apply_random_stabilizer(toric)
    A function that applies random stabilizers on qubit_matrix

## apply_n_independent_random_stabilizers(toric, n=int)
    A function that applies n random stabilizers on qubit_matrix

## apply_n_distinct_random_stabilizers(toric, n=int)
    A function that applies n random stabilizers on qubit_matrix
   
## error_ratio(qubit_matrix_current, qubit_matrix_next, p=float)
    A function that computes and returns error ratio for accepted values

## update_error_configuration(qubit_matrix, size, p)
    A function that updates error values in qubit_matrix according to error_ratio

## init_error(toric, qubit_matrix)
    A function that initializes individual errors (not in use)
