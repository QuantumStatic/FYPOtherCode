from __future__ import annotations
from math import sqrt, gcd, floor, ceil
from typing import Generator, NamedTuple, Union
from myfunctions import execute_this, nearMatching, compare_these
from statistics import mean
from functools import lru_cache
from collections import Counter


def polynomial_eval(polynomial, value):
    evaluation = int()
    for coefficient, power in enumerate(polynomial):
        pow_calc = int(1)
        for _ in range(power):
            pow_calc *= value
        evaluation += coefficient * pow_calc
    return (evaluation)


def horner_itr(polynomial, value):
    # bottoms-up implementation
    ''' horner algorithms aims to remove the dependency on the pow function or calculation the traditional 
        way by breaking it into self similar pieces (in a way breaking into a puzzle made up of itself) '''

    '''y = a(k) + y*x is the horner algorithm. where k ranges from n to 0. in the first iteration y just equals to the 
    innermost constant and nothing else. That is an implication of the self refrential equation not a planned occurence  '''

    summation = int()
    for index in reversed(polynomial):
        summation = index + value * summation

    return (summation)


def horner_recur(polynomial, value):
    # top-down implementation

    if len(polynomial) == 1:
        return polynomial.pop()

    coeff = polynomial.pop(0)
    return (coeff + value * horner_recur(polynomial, value))


def horner_alt(polynomial, value):

    def power(a, b):
        final = 0
        while (b):
            if b & 1:
                final *= a
            a *= a
            b //= 2
        return final

    evaluation = int()
    for coefficient, powe in enumerate(polynomial):
        evaluation += coefficient * power(value, powe)
    return (evaluation)


def fibPhi(n):
    phi = (1 + sqrt(5)) / 2
    return floor(pow(phi, n)/sqrt(5)+0.5)


@lru_cache(None)
def fibCache(n):
    if n < 2:
        return n
    return fibCache(n - 1) + fibCache(n - 2)


def dynamicFib(n):
    register = dict()
    register[0], register[1] = 0, 1

    def fib(n):
        try:
            return register[n]
        except KeyError:
            register[n] = fib(n-1) + fib(n-2)
            return register[n]

    return fib(n)


def fibMatrixExp(n):
    # any fibonacci number at nth postition can be written as as (n-1)th power of the matrix [[1,1][1,0]] mulitplied by [fibonaaci(1), fibonacci(0)]
    result = tuple([[1, 0], [0, 1]])
    OG = tuple([[1, 1], [1, 0]])
    n -= 1

    def matrixMultiply(a, b):
        fin = tuple([[0, 0], [0, 0]])
        fin[0][0] = (a[0][0]*b[0][0] + a[0][1]*b[1][0])  # % 10
        fin[0][1] = (a[0][0]*b[0][1] + a[0][1]*b[1][1])  # % 10
        fin[1][0] = (a[1][0]*b[0][0] + a[1][1]*b[1][0])  # % 10
        fin[1][1] = (a[1][0]*b[0][1] + a[1][1]*b[1][1])  # % 10
        return fin

    while n:
        if n & 1:
            result = matrixMultiply(OG, result)
        OG = matrixMultiply(OG, OG)
        n //= 2

    return matrixMultiply(result, tuple([[1, 0], [0, 1]]))[0][0]


def find_max_sub_array_recur(arrey):

    # this is found by running brute force against the recursive implementation. Depends on hardware specifications.
    recur_better_than_brute_point = 8

    def find_max_sub_array_brute(arrey):
        maxSum, start, end = float("-Inf"), int(), int()
        for x in range(len(arrey)):
            currSum = int()
            for y in range(x, len(arrey)):
                currSum += arrey[y]
                if currSum > maxSum:
                    start, end, maxSum = x, y, currSum
        return (start, end, maxSum)

    def find_Max_Mid(arr, begin, mid, end):
        leftSum, currSum, maxLeft = float("-Inf"), int(), int()
        for x in range(mid, begin - 1, -1):
            currSum += arr[x]
            if currSum > leftSum:
                leftSum = currSum
                maxLeft = x
        rightSum, currSum, maxRight = float("-Inf"), int(), int()
        for x in range(mid+1, end + 1):
            currSum += arr[x]
            if currSum > rightSum:
                rightSum = currSum
                maxRight = x

        return (maxLeft, maxRight, leftSum + rightSum)

    def find_Max(arr, begining, end):
        if begining is end:
            return (begining, end, arr[end])
        elif begining - end > recur_better_than_brute_point:
            mid = (begining + end) // 2
            leftBegin, leftEnd, leftSum = find_Max(arr, begining, mid)
            rightBegin, rightEnd, rightSum = find_Max(arr, mid + 1, end)
            midBegin, midEnd, midSum = find_Max_Mid(arr, begining, mid, end)
            if leftSum >= rightSum and leftSum >= midSum:
                return (leftBegin, leftEnd, leftSum)
            elif leftSum < rightSum and rightSum >= midSum:
                return (rightBegin, rightEnd, rightSum)
            else:
                return (midBegin, midEnd, midSum)
        else:
            return find_max_sub_array_brute(arrey)

    return find_Max(arrey, 0, len(arrey) - 1)


def find_max_sub_array_linear(arrey):
    ''' for a better understanding of this algorthim, plot y = f(x), where f(x) is defined as sum of all 
    elements of a list from x =0 to x =n. This algorthim finds the area under the graph when f(n) >= 0 '''

    maxSum, currSum, begin, end = float("-Inf"), float("-Inf"), int(), int()
    for x, val in enumerate(arrey):
        currEnd = x
        if currSum > 0:
            currSum += val
        else:
            currBegin, currSum = x, val
        if currSum > maxSum:
            maxSum, begin, end = currSum, currBegin, currEnd

    return (begin, end, maxSum)


def extractMaxMin(arg):
    # Following code does 3 comparisions for every 2 elements, rather than intutive way of 2 comparisions per element
    n, Max, Min = len(arg), int(), int()
    if n % 2:
        Max = Min = arg[0]
        for x in range(1, n-1, 2):
            big, SMALL, n1, n2 = int(), int(), arg[x], arg[x+1]
            if n1 > n2:
                big, SMALL = n1, n2
            else:
                big, SMALL = n2, n1
            if big > Max:
                Max = big
            if SMALL < Min:
                Min = SMALL
    else:
        Max = Min = float("-inf")
        for x in range(0, n-1, 2):
            big, SMALL, n1, n2 = int(), int(), arg[x], arg[x+1]
            if n1 > n2:
                big, SMALL = n1, n2
            else:
                big, SMALL = n2, n1
            if big > Max:
                Max = big
            if SMALL < Min:
                Min = SMALL


def IorderSelection(List, order):

    def Partition(List, lowEnd, highEnd):
        leftIndex, rightIndex, tempList = lowEnd, highEnd - \
            1, List[lowEnd:highEnd]
        pivot = List[nearMatching(tuple(tempList), mean(tempList))+lowEnd]
        while True:
            while rightIndex >= lowEnd and List[rightIndex] >= pivot:
                rightIndex -= 1
            while leftIndex < highEnd and List[leftIndex] <= pivot:
                leftIndex += 1
            if leftIndex < rightIndex:
                List[leftIndex], List[rightIndex] = List[rightIndex], List[leftIndex]
            else:
                return rightIndex

    def Select(List, lowEnd, highEnd):
        if lowEnd < highEnd:
            positionedElement = Partition(List, lowEnd, highEnd)
            if positionedElement == order:
                return List[positionedElement]
            elif order < positionedElement:
                return Select(List, lowEnd, positionedElement)
            else:
                return Select(List, positionedElement+1, highEnd)
        return

    return Select(List, 0, len(List))


def longestCommonSubsequence(seq1: str, seq2: str, only_length=False):
    seq1_len, seq2_len = len(seq1), len(seq2)
    matching_sequence_table = list()

    if not only_length:
        for _ in range(seq1_len + 1):
            matching_sequence_table.append([0] * (seq2_len + 1))

        for x in range(1, seq1_len + 1):
            for y in range(1, seq2_len + 1):
                if seq1[x-1] == seq2[y-1]:
                    matching_sequence_table[x][y] = 1 + \
                        matching_sequence_table[x-1][y-1]
                else:
                    matching_sequence_table[x][y] = max(
                        matching_sequence_table[x-1][y], matching_sequence_table[x][y-1])

        LCS_len = matching_sequence_table[seq1_len][seq2_len]
        longest_common_subsequence = [0] * LCS_len
        x, y = seq1_len, seq2_len
        while matching_sequence_table[x][y] != 0:
            if matching_sequence_table[x][y - 1] != matching_sequence_table[x][y]:
                longest_common_subsequence[LCS_len - 1] = seq2[y - 1]
                x -= 1
                LCS_len -= 1
            y -= 1

        return ''.join(longest_common_subsequence)

    else:
        for _ in range(2):
            matching_sequence_table.append([0] * (seq2_len + 1))

        for _ in range(seq1_len):
            for y in range(1, seq2_len + 1):
                if seq1[0] == seq2[y-1]:
                    matching_sequence_table[1][y] = 1 + \
                        matching_sequence_table[0][y-1]
                else:
                    matching_sequence_table[1][y] = max(
                        matching_sequence_table[0][y], matching_sequence_table[1][y-1])
            seq1 = seq1[1:]
            matching_sequence_table[0] = matching_sequence_table[1]

        return matching_sequence_table[1][seq2_len]


def binaryExponentiation(a: int, b: int, modulo: int = None):

    result = 1
    if modulo is None:
        modulo = a + 1

    while b:
        if b & 1:
            result *= a
            result %= modulo
        a *= a
        a %= modulo
        b //= 2

    return result


def jugProblem(jug1: int, jug2: int, toMeasure: int, justSteps=False):
    smallerJugSize, biggerJugSize, jugToFill = jug1, jug2, toMeasure

    if jugToFill % gcd(smallerJugSize, biggerJugSize) != 0:
        print("No Solution")
        return

    if smallerJugSize > biggerJugSize:
        smallerJugSize, biggerJugSize = biggerJugSize, smallerJugSize

    steps, steps_taken = int(), list()
    smallJug = bigJug = int()
    while bigJug != jugToFill and smallJug != jugToFill:
        if bigJug == 0:
            bigJug = biggerJugSize
        elif smallJug == smallerJugSize:
            smallJug = 0
        else:
            spaceInSmallerJug = smallerJugSize - smallJug
            smallJug, bigJug = smallJug + (spaceInSmallerJug if bigJug >= spaceInSmallerJug else bigJug), bigJug - (
                spaceInSmallerJug if bigJug >= spaceInSmallerJug else bigJug)
        steps_taken.append((smallJug, bigJug))
        steps += 1

    if justSteps:
        return steps
    else:
        return steps_taken


def constructionProblem(toConstruct: int, pieces: tuple, total_ways: bool = False, minimum_pieces: bool = True, sequence: bool = False) -> int:
    constructionTable = list()
    nOfPieces = len(pieces)

    if total_ways:
        for _ in range(2):
            constructionTable.append(list())

        if pieces[0] == 0:
            constructionTable[0].append(1)
            constructionTable[0].extend([0 for _ in range(toConstruct)])
        else:
            for x in range(toConstruct+1):
                constructionTable[0].append(1 if x % pieces[0] == 0 else 0)
        constructionTable[1].extend([0 for _ in range(toConstruct+1)])

        pieces = pieces[1:]

        for _ in range(nOfPieces-1):
            for y in range(toConstruct+1):
                ways = constructionTable[0][y]
                if pieces[0] <= y:
                    ways += constructionTable[1][y - pieces[0]]
                constructionTable[1][y] = ways
            pieces = pieces[1:]
            constructionTable[0] = constructionTable[1]

        return constructionTable[1][toConstruct]

    else:
        if sequence:
            for _ in range(nOfPieces):
                constructionTable.append(list())

            if pieces[0] == 0:
                constructionTable[0].append(1)
                constructionTable[0].extend([0 for _ in range(toConstruct)])
            else:
                for x in range(toConstruct+1):
                    constructionTable[0].append(
                        x // pieces[0] if x % pieces[0] == 0 else 0)

            for x in range(1, nOfPieces):
                for y in range(toConstruct+1):
                    ways = constructionTable[x-1][y]
                    if pieces[x] <= y:
                        ways = min(
                            1 + constructionTable[x][y - pieces[x]], ways)
                    constructionTable[x].append(ways)

            x = nOfPieces - 1
            y = toConstruct
            sequence = list()
            while x != 0:
                if constructionTable[x][y] != constructionTable[x-1][y]:
                    sequence.append(pieces[x])
                    y -= pieces[x]
                else:
                    x -= 1

            return tuple(sequence)

        else:
            for _ in range(2):
                constructionTable.append(list())

            if pieces[0] == 0:
                constructionTable[0].append(1)
                constructionTable[0].extend([0 for _ in range(toConstruct)])
            else:
                for x in range(toConstruct+1):
                    constructionTable[0].append(1 if x % pieces[0] == 0 else 0)
            constructionTable[1].extend([0 for _ in range(toConstruct+1)])

            pieces = pieces[1:]

            for _ in range(nOfPieces-1):
                for y in range(toConstruct+1):
                    ways = constructionTable[0][y]
                    if pieces[0] <= y:
                        ways = min(constructionTable[1][y - pieces[0]], ways)
                    constructionTable[1][y] = ways
                pieces = pieces[1:]
                constructionTable[0] = constructionTable[1]

            return constructionTable[1][toConstruct]


def primeGenerator(limit:int) -> list[int]:

    isprime, smallestPrimeFactor, primes = [False] * 2 + [True] * (limit - 2), [int()] * limit, list()
    primes_found = 0
    for i in range(2, limit):
        if isprime[i]:
            primes.append(i)
            # yield i
            smallestPrimeFactor[i] = i
            primes_found += 1
        j = 0
        while j < primes_found and i * primes[j] < limit and primes[j] <= smallestPrimeFactor[i]:
            isprime[i * primes[j]] = False
            smallestPrimeFactor[i * primes[j]] = primes[j]
            j += 1
    return primes


def primeGenerator_lazy(limit:int) -> Generator[int]:
    isprime, smallestPrimeFactor, primes = [False] * 2 + [True] * (limit - 2), [int()] * limit, list()
    primes_found = 0
    for i in range(2, limit):
        if isprime[i]:
            primes.append(i)
            yield i
            smallestPrimeFactor[i] = i
            primes_found += 1
        j = 0
        while j < primes_found and i * primes[j] < limit and primes[j] <= smallestPrimeFactor[i]:
            isprime[i * primes[j]] = False
            smallestPrimeFactor[i * primes[j]] = primes[j]
            j += 1


def prime_checker(suspected_prime):
    # Checking primes since '99. supports lists and individual numbers as well
    if isinstance(suspected_prime, list) or isinstance(suspected_prime, tuple):
        dummy = list()
        for prime_candidate in suspected_prime:
            dummy.append(prime_checker(prime_candidate))
        return dummy
    else:
        if suspected_prime < 300_000_000:
            return naive_prime_checker(suspected_prime)
        else:
            return miller_rabin_prime_checker(suspected_prime)


def naive_prime_checker(suspected_prime):
    suspected_prime = abs(suspected_prime)
    if suspected_prime == 1 or suspected_prime % 2 == 0 or suspected_prime % 3 == 0:
        return False
    end_point, prime_factor = ceil(suspected_prime**0.5), 5
    while end_point >= prime_factor:
        if suspected_prime % prime_factor == 0 or suspected_prime % (prime_factor + 2) == 0:
            return False
        prime_factor += 6
    return True


def miller_rabin_prime_checker(suspected_prime, k=40):
    def single_test(n, a):
        exp = n - 1
        while not exp & 1:
            exp >>= 1

        if pow(a, exp, n) == 1:
            return True

        while exp < n - 1:
            if pow(a, exp, n) == n - 1:
                return True
            exp <<= 1

        return False

    import random
    die = random.SystemRandom()

    for _ in range(k):
        witness = die.randrange(2, suspected_prime - 1)
        if not single_test(suspected_prime, witness):
            return False

    return True


def sieveErato(limit):
    # Sieve of Eratothenes. Looks up prime numbers upto almost 8 million in a second.
    if limit <= 90_000:
        primes, index, endPoint, result = [
            False, True] * (limit//2+1), 3, ceil(limit**0.5) + 1, [2]
        while index <= endPoint:  # sqrt of limit is the endpoint
            for compositeNum in range(index ** 2, limit + 1, index * 2):
                primes[compositeNum] = False
            index += 2
            while not primes[index]:
                index += 2
        for x in range(3, len(primes), 2):
            if primes[x]:
                result.append(x)
        return (result)
    else:
        primes = list()
        for x in range(limit):
            if prime_checker(x):
                primes.append(x)
        return primes


def prime_factoriser(n):
    # I am a rookie hence this implementation. upgrade due, feel free to suggest improvements

    if prime_checker(n):
        return ([n])
    prime_factor, list_of_factors = 2, list()
    while n % prime_factor == 0:
        n //= prime_factor
        list_of_factors.append(prime_factor)
    if n == 1:
        return (list_of_factors)
    end_point, prime_factor = ceil(sqrt(n)), 3
    while prime_factor < end_point+1:
        if n % prime_factor == 0:
            n //= prime_factor
            list_of_factors.append(prime_factor)
            if n == 1:
                return(list_of_factors)
        else:
            prime_factor += 2
    list_of_factors.append(n)
    return(list_of_factors)



def knapsack_01_dp(profits: list[int], weights: list[int], capacity: int):
    # Knapsack problem.
    # RETURNING THE CHOSEN OBJECTS IS BROKEN.

    if capacity == 0 or len(profits) == 0:
        return 0
    if len(profits) == 1:
        return profits[0] if weights[0] <= capacity else 0

    dp = [[0 for _ in range(capacity+1)] for _ in range(len(profits)+1)]

    object_to_carry = NamedTuple(
        'object_to_carry', [('profit', int), ('weight', int)])
    objects_to_carry: list[object_to_carry] = []

    for profit, weight in zip(profits, weights):
        objects_to_carry.append(object_to_carry(profit, weight))

    objects_to_carry.sort(key=lambda x: x.profit)
    objects_to_carry.sort(key=lambda x: x.weight)

    final_objects: list[tuple[int, int, object_to_carry]] = []

    for profit in range(1, len(profits)+1):
        for weight in range(1, capacity+1):
            if objects_to_carry[profit-1].weight <= weight:
                dp[profit][weight] = max(dp[profit-1][weight], dp[profit-1][weight -
                                         objects_to_carry[profit-1].weight] + objects_to_carry[profit-1].profit)
            else:
                dp[profit][weight] = dp[profit-1][weight]

    print(*dp, sep='\n')
    return final_objects


def longest_multiplication_subsequence(A: list[int]):

    # Counts occurrences of each number in the list.
    occurences = Counter(A)

    # removing duplicates from the list.
    A = tuple(set(number for number in A))

    max_chains: dict[int, list[int]] = {1: [1]}
    for i in range(len(A)):
        max_chain: list[int] = []
        # Checking if the number is a prime.
        if prime_checker(A[i]):
            max_chain = [1]*occurences.setdefault(1, 0)
        else:
            # Looping over numbers to see what elements divide A[i]
            j = 0
            while A[j] <= A[i] // 2:
                if A[i] % A[j] == 0:
                    curr_chain = max_chains.setdefault(A[j], [])
                    if len(curr_chain) > len(max_chain):
                        max_chain = curr_chain.copy()
                j += 1
        max_chain.extend([A[i]]*occurences[A[i]])

        # Storing the max chain for each number so that it can be used later.
        max_chains[A[i]] = max_chain

        # Finding the longest chain of numbers in max_chains
        longest_chains: list[list[int]] = []
        for key in max_chains:
            if not any(longest_chains) or len(max_chains[key]) > len(longest_chains[0]):
                longest_chains = [max_chains[key].copy()]
            elif len(max_chains[key]) == len(longest_chains[0]):
                longest_chains.append(max_chains[key].copy())

    return longest_chains


def heaviest_multiplication_subsequence(A: list[int]):

    # Counts occurrences of each number in the list.
    occurences = Counter(A)

    # removing duplicates from the list.
    A = tuple(set(number for number in A))

    heavy_chains: dict[int, list[int]] = {1: [1]}
    for i in range(len(A)):
        heavy_chain: list[int] = []
        # Checking if the number is a prime.
        if prime_checker(A[i]):
            heavy_chain = [1]*occurences.setdefault(1, 0)
        else:
            # Looping over numbers to see what elements divide A[i]
            j = 0
            while A[j] <= A[i] // 2:
                if A[i] % A[j] == 0:
                    curr_chain = heavy_chains.setdefault(A[j], [])
                    if sum(curr_chain) > sum(heavy_chain):
                        heavy_chain = curr_chain.copy()
                j += 1
        heavy_chain.extend([A[i]]*occurences[A[i]])

        # Storing the max chain for each number so that it can be used later.
        heavy_chains[A[i]] = heavy_chain

        # Finding the longest chain of numbers in max_chains
        heaviest_chains: list[list[int]] = []
        for key in heavy_chains:
            if not any(heaviest_chains) or sum(heavy_chains[key]) > sum(heaviest_chains[0]):
                heaviest_chains = [heavy_chains[key].copy()]
            elif sum(heavy_chains[key]) == sum(heaviest_chains[0]):
                heaviest_chains.append(heavy_chains[key].copy())

    return heaviest_chains


def weighted_scheduling_problem(wieghts: list[int], start: list[int], end: int):
    class Job:
        def __init__(self, weight: int, start: int, end: int, prev: int):
            self.weight = weight
            self.start = start
            self.end = end
            self.prev = prev

    jobs: list[Job] = []

    jobs: list[Job] = []
    for weight, start, end in zip(wieghts, start, end):
        jobs.append(Job(weight, start, end, 0))

    jobs.sort(key=lambda x: x.end)

    for x in range(len(jobs)):
        x_cpy = x - 1
        while x_cpy >= 0:
            if jobs[x].start >= jobs[x_cpy].end:
                jobs[x].prev = x_cpy + 1
                break
            x_cpy -= 1

    optimal_schedule: list[int] = [0]
    for x in range(len(jobs)):
        print(
            f"max(Optimal weight[{jobs[x].prev}] + Weights[{x}] = max({optimal_schedule[jobs[x].prev]} + {jobs[x].weight}, {optimal_schedule[x]})")
        optimal_schedule.append(
            max(jobs[x].weight + optimal_schedule[jobs[x].prev], optimal_schedule[x]))

    # print(*jobs)
    print(optimal_schedule)


def rabin_karp(text: str, pattern: str) -> Union[int, None]:

    len_pattern = len(pattern)

    # Hash of the pattern.
    def hash(pattern: str) -> int:
        nonlocal len_pattern
        hash_value = 0
        for index, value in enumerate(pattern):
            hash_value += ord(value) * pow(len_pattern, index)
        return hash_value

    def verify(text: str, pattern: str) -> bool:
        return text == pattern

    pattern_hash = hash(pattern)

    for x in range(len(text) - len_pattern + 1):
        if hash(text[x:x+len_pattern]) == pattern_hash:
            if verify(text[x:x+len_pattern], pattern):
                return x

    return False


def kmp_algorithm(text: str, pattern: str) -> Union[int, None]:

    prefix_table_element = NamedTuple(
        'prefix_table_element', [('value', str), ('index', int)])
    prefix_table: list[prefix_table_element] = [prefix_table_element(
        value=None, index=0), prefix_table_element(value=pattern[0], index=0)]
    x, y = 1, 1

    # prefix table construction
    while x < len(pattern):
        if pattern[x] == prefix_table[y].value:
            prefix_table.append(prefix_table_element(
                value=pattern[x], index=y))
            y += 1
        elif pattern[x] == prefix_table[1].value:
            prefix_table.append(prefix_table_element(
                value=pattern[x], index=1))
            y = 2
        else:
            prefix_table.append(prefix_table_element(
                value=pattern[x], index=0))
            y = 1
        x += 1

    y, x = 0, 0
    while x < len(text):
        if text[x] == prefix_table[y+1].value:
            y += 1
            if y == len(pattern):
                return x - len(pattern) + 1
        elif y != 0:
            y = prefix_table[y].index
            continue    # don't skip till y is changing.
        x += 1

    return False


def text_justification(text: Union[list[str], str], line_length: int = 28) -> tuple[int]:

    if isinstance(text, str):
        text = text.split()

    total_words = len(text)

    DP: list[int] = [float('inf')]*len(text)
    cut_locations: list[int] = [-1]*len(text)

    # score of a split
    def _badness(i: int, j: int) -> int:
        curr_len = sum(map(len, text[i:j])) + (j - i)
        if curr_len > line_length:
            return float('inf')
        return pow(line_length - curr_len, 3)

    def recursive_relation(i: int) -> int:
        if DP[i] == float('inf'):
            if total_words - i == 1:
                DP[i] = _badness(i, total_words)
                cut_locations[i] = total_words
            else:
                for j in range(i+1, total_words):
                    if (score_of_this_cut := recursive_relation(j) + _badness(i, j)) < DP[i]:
                        DP[i] = score_of_this_cut
                        cut_locations[i] = j

        return DP[i]

    recursive_relation(0)
    return cut_locations, DP


# @execute_this
def main():
    text = "Tushar roy likes to code".split()
    locations, table = text_justification(text, line_length=10)
    print(locations, table)
    for x in range(1, len(text)+1):
        print(text[x-1], end=' ')
        try:
            if locations[x] != locations[x-1]:
                print()
        except IndexError:
            print()
            break
