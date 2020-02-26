# MCMC
    This class implements the MCMC sampling algorithm for probability based reward scheme.
    Defines rule_table

## Chain
# __init__(self, size, p):
    self.toric innehåller ett toric objekt
    self.size håller reda på storlek för toric, behövs egentligen inte men smidigt då andra funktioner i denna klass kräver det
    self.p p för kedjan
    self.p_logical sannolikheten att göra logiska flippar i kedjan, bör vara 0 i alla kedjor utom den översta

# permute_error(self): # eventually rewrite to remove middle steps.
    wrapper funktion för att permutera toric

# plot(self, name):
    plot current state of chain

# set_p_logical(self, p_logical):
    ställ in p_logical, bör bara användas för översta kedjan

# get_p(self):
    returnera p

# get_qubit_matrix(self):
    returnera qubit_matrix för toric

# get_toric(self):
    returnera toricen själv

# set_toric(self, new_toric):
    sätt byt ut toric för aktuell chain, aktuellt vid flips

## r_flip(chain_lo, chain_hi):
    försök att göra en flip mellan två chains

## apply_random_logical(qubit_matrix, size=int)
    A function that applies a random logical operator and returns the new qubit_matrix

## apply_logical_vertical(qubit_matrix, size=int, col=int, operator=int)
    A function that applies a specified logical operator (X or Z) to specified column
    and returns the new qubit_matrix

## apply_logical_vertical(qubit_matrix, size=int, row=int, operator=int)
    A function that applies a specified logical operator (X or Z) to specified row
    and returns the new qubit_matrix

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
