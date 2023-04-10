from time import process_time_ns
import types
import statistics
from typing import Any
import concurrent.futures
from rich import print


def execution_time(total_time, show_time=True) -> str:
    # I get an OCD if it says 1 seconds and not 1 second hence this.

    ms, total_time = int(), total_time / 10**6
    mins = int(total_time // 60000)
    secs = int(total_time // 1000 - mins * 60)
    prefix, Min, Minz, Sec, Secx, Msec, Msecs, Mz, Mzs, suffix = "Execution Time: ", " Minute ", " Minutes ", " Second ", " Seconds ", " millisecond ", " milliseconds ", " microsecond ", " microseconds ", '\n'

    if total_time < 100:
        if total_time < 1:
            ms = round(total_time, 3)
        elif total_time < 10:
            ms = round(total_time, 2)
        else:
            ms = round(total_time, 1)
        if ms.is_integer():
            ms = int(ms)
    else:
        ms = int(round(total_time - (secs * 1000 + mins * 60000), 0))

    finalPrint = prefix
    if mins > 0:
        finalPrint += f"{mins}"
        if mins == 1:
            finalPrint += Min
        else:
            finalPrint += Minz
    if secs > 0:
        finalPrint += f"{secs}"
        if secs == 1:
            finalPrint += Sec
        else:
            finalPrint += Secx
    if ms > 0:
        if total_time < 1:
            ms = round(ms*100, 2)
            try:
                if ms.is_integer():
                    ms = int(ms)
            except:
                pass
            finalPrint += f"{ms}"
            if ms == 1:
                finalPrint += Mz
            else:
                finalPrint += Mzs
        else:
            finalPrint += f"{ms}"
            if ms == 1:
                finalPrint += Msec
            else:
                finalPrint += Msecs
    finalPrint += suffix
    if show_time:
        print(finalPrint)
    return finalPrint


def remove_duplicates_from(input_with_duplicates):
    # it's bad but it's mine, open to improvements

    if isinstance(input_with_duplicates, list):
        return(list(dict.fromkeys(input_with_duplicates)))
    else:
        return(combine_list(remove_duplicates_from(split_stuff(input_with_duplicates))))


def max_consecutive_elements_counter(array, element_to_be_checked):
    # old code, up for maintenance and improvement.
    consec_counter, final_answer = 0, 0
    for x in array:
        if x == element_to_be_checked:
            consec_counter += 1
        else:
            final_answer, consec_counter = (
                consec_counter if consec_counter > final_answer else final_answer), int()
    return (final_answer if final_answer > consec_counter else consec_counter)


def split_stuff(stuff, return_type=None):
    # too lazy to write this everytime.
    dummy_list = list()
    if isinstance(stuff, int):
        while stuff:
            dummy_list.insert(0, stuff % 10)
            stuff //= 10
    if isinstance(stuff, str):
        dummy_list = list(stuff)
    if return_type is not None:
        return (map(return_type, dummy_list))
    else:
        return (iter(dummy_list))


def combine_list(list_to_combine, lisIterType=None):
    # sometimes you just want elemnts of a list to be combined. It's not used often but does save lives when it matters
    if lisIterType is None:
        if isinstance(list_to_combine[0], str):
            string = str()
            return(string.join(list_to_combine))
        if isinstance(list_to_combine[0], int):
            num = int()
            for x in list_to_combine:
                num = num*10 + x
            return(num)
        if isinstance(list_to_combine[0], list) or isinstance(list_to_combine[0], tuple):
            dummy = list()
            for element in list_to_combine:
                dummy.append(combine_list(element))
            return (dummy)
    else:
        if isinstance(lisIterType, str):
            string = str()
            return(string.join(list_to_combine))
        if isinstance(lisIterType, int):
            num = int()
            for x in list_to_combine:
                num = num*10 + x
            return(num)
        if isinstance(lisIterType, list) or isinstance(lisIterType, tuple):
            dummy = list()
            for element in list_to_combine:
                dummy.append(combine_list(element))
            return (dummy)
        return False


def list_substractor(main_list, list_to_substract):
    # I don't remember half this file. I proabably was too lazy and just wrote it thinking that I might it need it later

    list_to_substract = set(list_to_substract)
    return(list(filter(lambda x: x not in list_to_substract, main_list)))


def find_pythogorean_triplets_till(num):
    # bad code. improvemnt known. will be updated soon to run in linear time

    list_of_triplets = list()
    for m in range(num+1):
        for n in range(m):
            a, b, c = m**2 - n**2, 2*m*n, m**2 + n**2
            list_of_triplets.extend([a, b, c])
    yield(list_of_triplets)


def check_list_containment(main_list, list_to_check, method1=None, method2=None):
    # set method removes duplicates before checking can be a pain someitmes, hence my code. looking for improvements
    if method1:
        return set(list_to_check).issubset(set(main_list))
    elif method2:
        listUnderScrutiny = iter(list_to_check)
        for x in listUnderScrutiny:
            if x in main_list:
                main_list.remove(x)
            else:
                return False
        return True
    else:
        raise Exception("No method found to be specified")


def clear_output_screen():
    from os import system
    system("clear") 


# pass general arguments in a single level depth iterable ie. list, tuple
PARAM, break_param = None, False

def execute_this(function) -> tuple:
    # I was tired of importing time and then making a start variable, calling the execution time function, I just wrote this
    clear_output_screen()
    global PARAM

    param_type = type(function)
    if param_type is not types.FunctionType and param_type is not types.BuiltinFunctionType and param_type is not types.BuiltinMethodType:
        PARAM = function
        return execute_this

    print(f"Executing {function.__name__}")
    start = process_time_ns()

    if PARAM is None:
        to_return = function()
    else:
        to_return = function(*PARAM)

    return (execution_time(process_time_ns()-start), to_return)


def compare_these(*functions, **function_args) -> tuple:

    times = []
    print_out = function_args.setdefault("output", False)

    def measure_run_time(function):
        def wrapper():
            nonlocal times, function_args, print_out
            args = function_args.setdefault(function.__name__, [])
            start = process_time_ns()
            output = function(*args)
            if output is not None and print_out:
                print(f" Function {function.__name__} gave the ouput {output}")
            timee = execution_time(process_time_ns()-start, show_time=False)
            print(" ", f"Ran {function.__name__} with time {timee}")

            times.append(timee)

        return wrapper

    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     tuple(map(lambda x: executor.submit(measure_run_time(x)), functions))

    for func in functions:
        measure_run_time(func)()

    return tuple(times)


def nearMatching(List, target):
    differneces = tuple(map(lambda x: abs(x - target), List))
    return differneces.index(min(differneces))


def run_cpp(path="/Users/utkarsh/Desktop/Utkarsh/Languages/C++/execute/main.cpp", timeIt=False, args: tuple = None, input: tuple = None):
    from subprocess import run, PIPE

    time_total = process_time_ns()
    p1 = run(["g++", "-o", "main", path], stderr=PIPE, text=True)
    time_total = process_time_ns() - time_total

    if p1.stderr.find("error") >= 0:
        print(p1.stderr)
        return

    to_run = ["./main"]
    if isinstance(args, list):
        to_run.extend(args)

    time_total -= process_time_ns()
    p1 = run(to_run, stdout=PIPE, stderr=PIPE, stdin=input, text=True)
    time_total += process_time_ns()

    if timeIt:
        execution_time(time_total)

    p1.stdout = tuple(str(p1.stdout).splitlines())
    p1.args = tuple(p1.args)

    return p1


def compare_outputs(*functions, **function_args):
    pass

def print_data_stats(data: list):
    print(f"Max: {max(data)}")
    print(f"Mean: {round(statistics.mean(data), 3)}")
    print(f"Median: {round(statistics.median(data), 3)}")
    print(f"Mode: {round(statistics.mode(data), 2)}")
    print(f"Variance: {round(statistics.variance(data), 3)}")
    print(f"Standard Deviation: {round(statistics.stdev(data), 3)}")