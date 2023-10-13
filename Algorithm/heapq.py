import heapq

def cookies(k, A):
    # Write your code here

    """
    1. Use heapq
    2. Heapify A to sort in place and maintain sort position
    For heappush
    3. Perform operation and append to heap with heappush to keep
    heap sorted in place
    4. Break loop and return count when first item in array is > k
    Or if length of A is greater than 1
    5. Return count if first item in A is >= k, else no valid solution
    Return -1
    """
    count = 0
    heapq.heapify(A)
    while A[0] < k and len(A) > 1:
        sw = heapq.heappop(A) + 2 * heapq.heappop(A)
        heapq.heappush(A, sw)
        count += 1
    return count if A[0] >= k else -1