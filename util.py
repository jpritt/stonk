
def read_safe(n, s, e):
    num_retries = 5
    for attempt_no in range(num_retries):
        try:
            data = pdr.get_data_yahoo(n, s, e)
            return data
        except:
            if attempt_no < (num_retries - 1):
                print("Error: Failed reading %s" % n)
            else:
                exit()