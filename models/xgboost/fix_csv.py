with open("X_test.csv", "r") as f_in:
    with open("X_test_fixed.csv", "w") as f_out:
        for line in f_in:
            f_out.write(",".join(line.split(",")[1:]))
            
