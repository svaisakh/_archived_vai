def print_progress(iteration, total, start_time, print_every=1e-2):
    progress = (iteration + 1) / total
    if iteration == total - 1:
        print("Completed in {}s.\n".format(int(time() - start_time)))
    elif (iteration + 1)  % max(1, int(total * print_every / 100)) == 0:
        print("{:.2f}% completed. Time - {}s, ETA - {}s\t\t".format(np.round(progress * 100, 2), int(time() - start_time), int((1 / progress - 1) * (time() - start_time))), end='\r', flush=True)

def dict_to_list(dictionary, idx_list_length=0, none_key='None'):
    for key, value in dictionary.items():
        if len(value) == 0:
            none_key = key
        
        if idx_list_length == 0:
            for index in value:
                if index >= idx_list_length:
                    idx_list_length = index + 1
                
    idx_list = [none_key] * idx_list_length

    for key, value in dictionary.items():
        for index in value:
            idx_list[index] = key
            
    return idx_list
