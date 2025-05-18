document.addEventListener('DOMContentLoaded', () => {
    const questionsListDiv = document.getElementById('questions-list');
    const hintModal = document.getElementById('hint-modal');
    const hintQuestionTitle = document.getElementById('hint-question');
    const hintContentDiv = document.getElementById('hint-content');
    const closeBtn = document.querySelector('.close-btn');

    // Complete list of Fasal coding questions with hints and answers
    const fasalQuestions = [
        // ... (your existing question array remains the same)
        {

            "question": "Write a function to find the second largest element in an array of integers.",
    "description": "This question tests your ability to iterate through an array and keep track of multiple extreme values.",
    "hint": "Maintain two variables: one for the largest and one for the second largest. Iterate through the array, updating these variables accordingly.",
    "answer": `
      <p><strong>Sample Answer (Python):</strong></p>
      <pre><code class="language-python">def find_second_largest(nums):
    if len(nums) < 2:
        return None
    largest = float('-inf')
    second_largest = float('-inf')
    for num in nums:
        if num > largest:
            second_largest = largest
            largest = num
        elif num > second_largest and num != largest:
            second_largest = num
    if second_largest == float('-inf'):
        return None
    return second_largest
      </code></pre>
      <p><strong>Sample Answer (Java):</strong></p>
      <pre><code class="language-java">public class SecondLargest {
    public static Integer findSecondLargest(int[] nums) {
        if (nums == null || nums.length < 2) {
            return null;
        }
        int largest = Integer.MIN_VALUE;
        int secondLargest = Integer.MIN_VALUE;
        for (int num : nums) {
            if (num > largest) {
                secondLargest = largest;
                largest = num;
            } else if (num > secondLargest && num != largest) {
                secondLargest = num;
            }
        }
        if (secondLargest == Integer.MIN_VALUE) {
            return null;
        }
        return secondLargest;
    }
}
      </code></pre>
      <p><strong>Key Aspects to Highlight:</strong> Handling edge cases (array length less than 2), initializing variables appropriately, iterating through the array, updating largest and second largest based on comparisons, handling cases with duplicate largest elements, time complexity (O(n)).</p>
    `
  },
  {
    "question": "Write a function to check if a given string is a valid shuffle of two other strings.",
    "description": "This question tests your ability to compare string compositions and lengths.",
    "hint": "The length of the shuffled string must be equal to the sum of the lengths of the two original strings. Count the frequency of each character in all three strings and compare.",
    "answer": `
      <p><strong>Sample Answer (Python):</strong></p>
      <pre><code class="language-python">from collections import Counter

def is_valid_shuffle(s1, s2, shuffled):
    if len(shuffled) != len(s1) + len(s2):
        return False
    return Counter(s1) + Counter(s2) == Counter(shuffled)
      </code></pre>
      <p><strong>Sample Answer (Java):</strong></p>
      <pre><code class="language-java">import java.util.HashMap;
import java.util.Map;

public class ValidShuffle {
    public static boolean isValidShuffle(String s1, String s2, String shuffled) {
        if (shuffled.length() != s1.length() + s2.length()) {
            return false;
        }
        Map<Character, Integer> counts = new HashMap<>();
        for (char c : s1.toCharArray()) {
            counts.put(c, counts.getOrDefault(c, 0) + 1);
        }
        for (char c : s2.toCharArray()) {
            counts.put(c, counts.getOrDefault(c, 0) + 1);
        }
        Map<Character, Integer> shuffledCounts = new HashMap<>();
        for (char c : shuffled.toCharArray()) {
            shuffledCounts.put(c, shuffledCounts.getOrDefault(c, 0) + 1);
        }
        return counts.equals(shuffledCounts);
    }
}
      </code></pre>
      <p><strong>Key Aspects to Highlight:</strong> Checking the lengths first, using a dictionary/map (or Counter in Python) to store character frequencies, comparing the combined frequencies of the original strings with the shuffled string's frequencies, time complexity (O(m+n+k) where m, n, and k are lengths of the strings).</p>
    `
  },


  {
    "question": "Write a function to implement a basic stack that supports the 'min' operation in O(1) time.",
    "description": "This question tests your understanding of stack operations and efficient tracking of the minimum element.",
    "hint": "Maintain an auxiliary stack that stores the minimum element encountered so far at each step.",
    "answer": `
      <p><strong>Sample Answer (Python):</strong></p>
      <pre><code class="language-python">class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val):
        self.stack.append(val)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)

    def pop(self):
        if self.stack:
            popped = self.stack.pop()
            if self.min_stack and popped == self.min_stack[-1]:
                self.min_stack.pop()
            return popped
        return None

    def top(self):
        return self.stack[-1] if self.stack else None

    def getMin(self):
        return self.min_stack[-1] if self.min_stack else None
      </code></pre>
      <p><strong>Sample Answer (Java):</strong></p>
      <pre><code class="language-java">import java.util.Stack;

public class MinStack {
    private Stack<Integer> stack = new Stack<>();
    private Stack<Integer> minStack = new Stack<>();

    public void push(int val) {
        stack.push(val);
        if (minStack.isEmpty() || val <= minStack.peek()) {
            minStack.push(val);
        }
    }

    public Integer pop() {
        if (!stack.isEmpty()) {
            int popped = stack.pop();
            if (!minStack.isEmpty() && popped == minStack.peek()) {
                minStack.pop();
            }
            return popped;
        }
        return null;
    }

    public Integer top() {
        return stack.isEmpty() ? null : stack.peek();
    }

    public Integer getMin() {
        return minStack.isEmpty() ? null : minStack.peek();
    }
}
      </code></pre>
      <p><strong>Key Aspects to Highlight:</strong> Using two stacks, the logic for pushing onto the min stack (only if the current value is less than or equal to the top of the min stack), the logic for popping from the min stack (only if the popped value from the main stack is equal to the top of the min stack), O(1) time complexity for all operations.</p>
    `
  },
  {
    "question": "Write a function to check if a given linked list is a palindrome.",
    "description": "This question tests your understanding of linked list traversal and comparison.",
    "hint": "One approach is to reverse the second half of the linked list and compare it with the first half. Another approach involves using a stack.",
    "answer": `
      <p><strong>Sample Answer (Python - Using Stack):</strong></p>
      <pre><code class="language-python">class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def is_palindrome_linked_list(head):
    if not head or not head.next:
        return True
    stack = []
    slow = head
    fast = head
    while fast and fast.next:
        stack.append(slow.val)
        slow = slow.next
        fast = fast.next.next
    if fast: # Odd length list, skip the middle element
        slow = slow.next
    while slow:
        if not stack or stack.pop() != slow.val:
            return False
        slow = slow.next
    return not stack
      </code></pre>
      <p><strong>Sample Answer (Java - Reversing Second Half):</strong></p>
      <pre><code class="language-java">public class PalindromeLinkedList {
    public static class ListNode {
        int val;
        ListNode next;
        ListNode() {}
        ListNode(int val) { this.val = val; }
        ListNode(int val, ListNode next) { this.val = val; this.next = next; }
    }

    public static boolean isPalindrome(ListNode head) {
        if (head == null || head.next == null) {
            return true;
        }
        ListNode slow = head;
        ListNode fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode secondHalf = reverseList(slow);
        ListNode firstHalf = head;
        while (secondHalf != null) {
            if (firstHalf.val != secondHalf.val) {
                return false;
            }
            firstHalf = firstHalf.next;
            secondHalf = secondHalf.next;
        }
        return true;
    }

    private static ListNode reverseList(ListNode head) {
        ListNode prev = null;
        ListNode current = head;
        while (current != null) {
            ListNode nextNode = current.next;
            current.next = prev;
            prev = current;
            current = nextNode;
        }
        return prev;
    }
}
      </code></pre>
      <p><strong>Key Aspects to Highlight:</strong> Understanding linked list traversal, using either a stack or reversing the second half for comparison, handling both even and odd length lists, time complexity (O(n)), space complexity (O(n) for stack approach, O(1) for reversing second half approach).</p>
    `
  },
  {
    "question": "Write a function to rotate an array of 'n' elements to the right by 'k' steps.",
    "description": "This question tests your ability to manipulate arrays efficiently with a specific rotation.",
    "hint": "Consider rotating the array in three steps: reverse the entire array, reverse the first 'k' elements, and reverse the remaining 'n-k' elements.",
    "answer": `
      <p><strong>Sample Answer (Python):</strong></p>
      <pre><code class="language-python">def rotate_array(nums, k):
    n = len(nums)
    k %= n  # Handle cases where k > n

    def reverse(arr, start, end):
        while start < end:
            arr[start], arr[end] = arr[end], arr[start]
            start += 1
            end -= 1

    reverse(nums, 0, n - 1)
    reverse(nums, 0, k - 1)
    reverse(nums, k, n - 1)
    return nums
      </code></pre>
      <p><strong>Sample Answer (Java):</strong></p>
      <pre><code class="language-java">public class RotateArray {
    public static void rotateArray(int[] nums, int k) {
        int n = nums.length;
        k %= n;

        reverse(nums, 0, n - 1);
        reverse(nums, 0, k - 1);
        reverse(nums, k, n - 1);
    }

    private static void reverse(int[] arr, int start, int end) {
        while (start < end) {
            int temp = arr[start];
            arr[start] = arr[end];
            arr[end] = temp;
            start++;
            end--;
        }
    }
}
      </code></pre>
      <p><strong>Key Aspects to Highlight:</strong> Handling the case where k is greater than n using the modulo operator, implementing a helper function for reversing a subarray, performing the three reversals, time complexity (O(n)), space complexity (O(1) in-place).</p>
    `
  },
  {
    "question": "Write a function to check if a given string contains only digits.",
    "description": "This question tests your ability to iterate through a string and check character properties.",
    "hint": "Iterate through the string and check if each character is a digit using built-in functions or by comparing ASCII values.",
    "answer": `
      <p><strong>Sample Answer (Python):</strong></p>
      <pre><code class="language-python">def contains_only_digits(s):
    return s.isdigit()
      </code></pre>
      <p><strong>Sample Answer (Java):</strong></p>
      <pre><code class="language-java">public class ContainsDigits {
    public static boolean containsOnlyDigits(String s) {
        if (s == null || s.isEmpty()) {
            return false; // Or handle empty string as needed
        }
        for (char c : s.toCharArray()) {
            if (!Character.isDigit(c)) {
                return false;
            }
        }
        return true;
    }
}
      </code></pre>
      <p><strong>Key Aspects to Highlight:</strong> Using built-in string/character functions if allowed, iterating through the string and checking each character, handling empty or null strings, time complexity (O(n)).</p>
    `
  },
  {
    "question": "Write a function to find the intersection point of two singly linked lists. Assume the lists might or might not intersect.",
    "description": "This question tests your understanding of linked list traversal and finding a common node.",
    "hint": "One approach is to find the lengths of both lists, move the pointer of the longer list ahead by the difference in lengths, and then move both pointers simultaneously until they meet.",
    "answer": `
      <p><strong>Sample Answer (Python):</strong></p>
      <pre><code class="language-python">class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def get_intersection_node(headA, headB):
    lenA, lenB = 0, 0
    currA, currB = headA, headB
    while currA:
        lenA += 1
        currA = currA.next
    while currB:
        lenB += 1
        currB = currB.next

    currA, currB = headA, headB
    if lenA > lenB:
        for _ in range(lenA - lenB):
            currA = currA.next
    elif lenB > lenA:
        for _ in range(lenB - lenA):
            currB = currB.next

    while currA and currB and currA != currB:
        currA = currA.next
        currB = currB.next

    return currA
      </code></pre>
      <p><strong>Sample Answer (Java):</strong></p>
      <pre><code class="language-java">public class IntersectionLinkedList {
    public static class ListNode {
        int val;
        ListNode next;
        ListNode() {}
        ListNode(int val) { this.val = val; }
        ListNode(int val, ListNode next) { this.val = val; this.next = next; }
    }

    public static ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        int lenA = 0, lenB = 0;
        ListNode currA = headA, currB = headB;
        while (currA != null) {
            lenA++;
            currA = currA.next;
        }
        while (currB != null) {
            lenB++;
            currB = currB.next;
        }

        currA = headA;
        currB = headB;
        if (lenA > lenB) {
            for (int i = 0; i < lenA - lenB; i++) {
                currA = currA.next;
            }
        } else {
            for (int i = 0; i < lenB - lenA; i++) {
                currB = currB.next;
            }
        }

        while (currA != null && currB != null && currA != currB) {
            currA = currA.next;
            currB = currB.next;
        }
        return currA;
    }
}
      </code></pre>
      <p><strong>Key Aspects to Highlight:</strong> Finding the lengths of both lists, moving the pointer of the longer list, moving both pointers simultaneously, returning the intersection node (or null if no intersection), time complexity (O(m+n)), space complexity (O(1)).</p>
    `
  },
  {
    "question": "Write a function to perform a level order traversal (Breadth-First Search) of a binary tree and print the nodes at each level.",
    "description": "This question tests your understanding of tree traversal using a queue.",
    "hint": "Use a queue to store nodes at the current level. Process the current level, print the nodes, and enqueue their children for the next level.",
    "answer": `
      <p><strong>Sample Answer (Python):</strong></p>
      <pre><code class="language-python">from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def level_order_traversal(root):
    if not root:
        return
    queue = deque([root])
    while queue:
        level_size = len(queue)
        current_level = []
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        print(current_level)
      </code></pre>
      <p><strong>Sample Answer (Java):</strong></p>
      <pre><code class="language-java">import java.util.LinkedList;
import java.util.Queue;
import java.util.ArrayList;
import java.util.List;

public class LevelOrderTraversal {
    public static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode() {}
        TreeNode(int val) { this.val = val; }
        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }

    public static void levelOrder(TreeNode root) {
        if (root == null) {
            return;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int levelSize = queue.size();
            List<Integer> currentLevel = new ArrayList<>();
            for (int i = 0; i < levelSize; i++) {
                TreeNode node = queue.poll();
                currentLevel.add(node.val);
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
            }
            System.out.println(currentLevel);
        }
    }
}
      </code></pre>
      <p><strong>Key Aspects to Highlight:</strong> Using a queue, processing level by level, keeping track of the number of nodes at the current level, enqueuing children for the next level, time complexity (O(n)), space complexity (O(w) where w is the maximum width of the tree).</p>
    `
  },
  {
    "question": "Write a function to implement a basic selection sort algorithm.",
    "description": "This question tests your understanding of a simple sorting algorithm.",
    "hint": "Iterate through the array, find the minimum element in the unsorted part, and swap it with the element at the beginning of the unsorted part.",
    "answer": `
      <p><strong>Sample Answer (Python):</strong></p>
      <pre><code class="language-python">def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_index = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_index]:
                min_index = j
        arr[i], arr[min_index] = arr[min_index], arr[i]
    return arr
      </code></pre>
      <p><strong>Sample Answer (Java):</strong></p>
      <pre><code class="language-java">public class SelectionSort {
    public static void selectionSort(int[] arr) {
        int n = arr.length;
        for (int i = 0; i < n - 1; i++) {
            int minIndex = i;
            for (int j = i + 1; j < n; j++) {
                if (arr[j] < arr[minIndex]) {
                    minIndex = j;
                }
            }
            int temp = arr[minIndex];
            arr[minIndex] = arr[i];
            arr[i] = temp;
        }
    }
}
      </code></pre>
      <p><strong>Key Aspects to Highlight:</strong> Outer loop iterating through the array, inner loop finding the minimum element in the unsorted part, swapping the minimum element with the current position, time complexity (O(n^2)).</p>
    `
  },

  {
    "question": "Write a function to implement a basic Insertion Sort algorithm.",
    "description": "This question tests your understanding of another simple sorting algorithm that builds the final sorted array one item at a time.",
    "hint": "Iterate through the array. For each element, compare it with the previous elements and insert it into the correct sorted position.",
    "answer": `
      <p><strong>Sample Answer (Python):</strong></p>
      <pre><code class="language-python">def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr
      </code></pre>
      <p><strong>Sample Answer (Java):</strong></p>
      <pre><code class="language-java">public class InsertionSort {
    public static void insertionSort(int[] arr) {
        int n = arr.length;
        for (int i = 1; i < n; ++i) {
            int key = arr[i];
            int j = i - 1;
            while (j >= 0 && key < arr[j]) {
                arr[j + 1] = arr[j];
                j = j - 1;
            }
            arr[j + 1] = key;
        }
    }
}
      </code></pre>
      <p><strong>Key Aspects to Highlight:</strong> Outer loop iterating from the second element, inner loop comparing the current element with the sorted part, shifting elements to make space for insertion, inserting the element in the correct position, time complexity (O(n^2) in the worst and average case, O(n) in the best case for an already sorted array).</p>
    `
  },
  {
    "question": "Write a function to find all subsets of a given set of distinct integers.",
    "description": "This question tests your ability to generate combinations or power sets.",
    "hint": "One way is to consider each element and decide whether to include it or exclude it in each subset. Another approach involves iteratively building subsets.",
    "answer": `
      <p><strong>Sample Answer (Python - Iterative):</strong></p>
      <pre><code class="language-python">def find_subsets(nums):
    subsets = [[]]
    for num in nums:
        new_subsets = []
        for subset in subsets:
            new_subsets.append(subset + [num])
        subsets.extend(new_subsets)
    return subsets
      </code></pre>
      <p><strong>Sample Answer (Java - Iterative):</strong></p>
      <pre><code class="language-java">import java.util.ArrayList;
import java.util.List;

public class FindSubsets {
    public static List<List<Integer>> findSubsets(int[] nums) {
        List<List<Integer>> subsets = new ArrayList<>();
        subsets.add(new ArrayList<>());
        for (int num : nums) {
            int n = subsets.size();
            for (int i = 0; i < n; i++) {
                List<Integer> subset = new ArrayList<>(subsets.get(i));
                subset.add(num);
                subsets.add(subset);
            }
        }
        return subsets;
    }
}
      </code></pre>
      <p><strong>Key Aspects to Highlight:</strong> Understanding the concept of subsets/power set, iterative approach of building subsets by considering each element, time complexity (O(2^n * n) because there are 2^n subsets and creating each can take O(n) time), space complexity (O(2^n * n) to store all subsets).</p>
    `
  },
  {
    "question": "Write a function to implement a basic queue using a singly linked list.",
    "description": "This question tests your understanding of queue operations and linked list manipulation.",
    "hint": "Maintain pointers to the head (for dequeue) and tail (for enqueue) of the linked list.",
    "answer": `
      <p><strong>Sample Answer (Python):</strong></p>
      <pre><code class="language-python">class Node:
    def __init__(self, val):
        self.val = val
        self.next = None

class QueueLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

    def enqueue(self, val):
        new_node = Node(val)
        if not self.head:
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            self.tail = new_node

    def dequeue(self):
        if not self.head:
            return None
        val = self.head.val
        self.head = self.head.next
        if not self.head:
            self.tail = None
        return val

    def is_empty(self):
        return not self.head

    def size(self):
        count = 0
        curr = self.head
        while curr:
            count += 1
            curr = curr.next
        return count
      </code></pre>
      <p><strong>Sample Answer (Java):</strong></p>
      <pre><code class="language-java">public class QueueLinkedList<T> {
    private static class Node<T> {
        T val;
        Node<T> next;
        Node(T val) { this.val = val; }
    }

    private Node<T> head;
    private Node<T> tail;

    public void enqueue(T val) {
        Node<T> newNode = new Node<>(val);
        if (head == null) {
            head = newNode;
            tail = newNode;
        } else {
            tail.next = newNode;
            tail = newNode;
        }
    }

    public T dequeue() {
        if (head == null) {
            return null;
        }
        T val = head.val;
        head = head.next;
        if (head == null) {
            tail = null;
        }
        return val;
    }

    public boolean isEmpty() {
        return head == null;
    }

    public int size() {
        int count = 0;
        Node<T> current = head;
        while (current != null) {
            count++;
            current = current.next;
        }
        return count;
    }
}
      </code></pre>
      <p><strong>Key Aspects to Highlight:</strong> Using a linked list structure, maintaining head and tail pointers, implementing enqueue (add to tail), dequeue (remove from head), isEmpty, and size operations, time complexity (O(1) for enqueue and dequeue), space complexity (O(n) to store the queue).</p>
    `
  },

  {
    "question": "Write a function to check if a given binary tree is balanced (the height difference between the left and right subtrees of any node does not exceed 1).",
    "description": "This question tests your understanding of tree properties and recursion.",
    "hint": "Use a recursive approach that calculates the height of each subtree and checks the balance condition at each node.",
    "answer": `
      <p><strong>Sample Answer (Python):</strong></p>
      <pre><code class="language-python">class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def is_balanced_tree(root):
    def get_height(node):
        if not node:
            return 0
        left_height = get_height(node.left)
        right_height = get_height(node.right)
        if left_height == -1 or right_height == -1 or abs(left_height - right_height) > 1:
            return -1
        return max(left_height, right_height) + 1

    return get_height(root) != -1
      </code></pre>
      <p><strong>Sample Answer (Java):</strong></p>
      <pre><code class="language-java">public class BalancedBinaryTree {
    public static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode() {}
        TreeNode(int val) { this.val = val; }
        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }

    public static boolean isBalanced(TreeNode root) {
        return getHeight(root) != -1;
    }

    private static int getHeight(TreeNode node) {
        if (node == null) {
            return 0;
        }
        int leftHeight = getHeight(node.left);
        if (leftHeight == -1) return -1;
        int rightHeight = getHeight(node.right);
        if (rightHeight == -1) return -1;
        if (Math.abs(leftHeight - rightHeight) > 1) {
            return -1;
        }
        return Math.max(leftHeight, rightHeight) + 1;
    }
}
      </pre>
      <p><strong>Key Aspects to Highlight:</strong> Recursive approach, calculating the height of left and right subtrees, checking the absolute difference in heights, returning a special value (-1) to indicate imbalance, time complexity (O(n)).</p>
    `
  },
  {
    "question": "Write a function to find the longest common prefix among an array of strings.",
    "description": "This question tests your ability to compare strings and find common starting substrings.",
    "hint": "Compare the first string with the rest of the strings character by character. Stop when a mismatch is found or the end of any string is reached.",
    "answer": `
      <p><strong>Sample Answer (Python):</strong></p>
      <pre><code class="language-python">def longest_common_prefix(strs):
    if not strs:
        return ""
    prefix = strs[0]
    for i in range(1, len(strs)):
        j = 0
        while j < len(prefix) and j < len(strs[i]) and prefix[j] == strs[i][j]:
            j += 1
        prefix = prefix[:j]
        if not prefix:
            break
    return prefix
      </code></pre>
      <p><strong>Sample Answer (Java):</strong></p>
      <pre><code class="language-java">public class LongestCommonPrefix {
    public static String longestCommonPrefix(String[] strs) {
        if (strs == null || strs.length == 0) {
            return "";
        }
        String prefix = strs[0];
        for (int i = 1; i < strs.length; i++) {
            int j = 0;
            while (j < prefix.length() && j < strs[i].length() && prefix.charAt(j) == strs[i].charAt(j)) {
                j++;
            }
            prefix = prefix.substring(0, j);
            if (prefix.isEmpty()) {
                break;
            }
        }
        return prefix;
    }
}
      </pre>
      <p><strong>Key Aspects to Highlight:</strong> Handling the empty array case, initializing the prefix with the first string, comparing the prefix with subsequent strings character by character, updating the prefix when a mismatch occurs, stopping early if the prefix becomes empty, time complexity (O(S) where S is the total number of characters across all strings in the worst case).</p>
    `
  },
  {
    "question": "Write a function to check if a given string is a valid palindrome, considering only alphanumeric characters and ignoring case.",
    "description": "This is a variation of the palindrome question with specific constraints on character consideration.",
    "hint": "Filter out non-alphanumeric characters and convert the string to lowercase before checking if it reads the same forwards and backward.",
    "answer": `
      <p><strong>Sample Answer (Python):</strong></p>
      <pre><code class="language-python">import re

def is_valid_alphanumeric_palindrome(s):
    processed_s = ''.join(filter(str.isalnum, s)).lower()
    return processed_s == processed_s[::-1]
      </code></pre>
      <p><strong>Sample Answer (Java):</strong></p>
      <pre><code class="language-java">public class ValidAlphanumericPalindrome {
    public static boolean isPalindrome(String s) {
        if (s == null || s.isEmpty()) {
            return true;
        }
        StringBuilder sb = new StringBuilder();
        for (char c : s.toCharArray()) {
            if (Character.isLetterOrDigit(c)) {
                sb.append(Character.toLowerCase(c));
            }
        }
        String processedS = sb.toString();
        int left = 0;
        int right = processedS.length() - 1;
        while (left < right) {
            if (processedS.charAt(left) != processedS.charAt(right)) {
                return false;
            }
            left++;
            right--;
        }
        return true;
    }
}
      </pre>
      <p><strong>Key Aspects to Highlight:</strong> Filtering non-alphanumeric characters, converting to lowercase, using string reversal or two pointers to check for palindrome, handling empty or null strings, time complexity (O(n)).</p>
    `
  },
  {
    "question": "Write a function to implement a basic Binary Search Tree (BST) with 'insert' and 'search' operations.",
    "description": "This question tests your understanding of the BST data structure and its fundamental operations.",
    "hint": "For insertion, traverse the tree to find the correct position based on the value to be inserted. For search, traverse the tree based on the target value, going left if it's smaller and right if it's larger.",
    "answer": `
      <p><strong>Sample Answer (Python):</strong></p>
      <pre><code class="language-python">class TreeNode:
    def __init__(self, val=None):
        self.val = val
        self.left = None
        self.right = None

class BST:
    def __init__(self):
        self.root = None

    def insert(self, val):
        if not self.root:
            self.root = TreeNode(val)
        else:
            self._insert(self.root, val)

    def _insert(self, node, val):
        if val < node.val:
            if node.left is None:
                node.left = TreeNode(val)
            else:
                self._insert(node.left, val)
        elif val > node.val:
            if node.right is None:
                node.right = TreeNode(val)
            else:
                self._insert(node.right, val)

    def search(self, val):
        return self._search(self.root, val)

    def _search(self, node, val):
        if not node:
            return False
        if val == node.val:
            return True
        elif val < node.val:
            return self._search(node.left, val)
        else:
            return self._search(node.right, val)
      </code></pre>
      <p><strong>Sample Answer (Java):</strong></p>
      <pre><code class="language-java">public class BST {
    public static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode(int val) { this.val = val; }
    }

    private TreeNode root;

    public void insert(int val) {
        root = insertRecursive(root, val);
    }

    private TreeNode insertRecursive(TreeNode root, int val) {
        if (root == null) {
            return new TreeNode(val);
        }
        if (val < root.val) {
            root.left = insertRecursive(root.left, val);
        } else if (val > root.val) {
            root.right = insertRecursive(root.right, val);
        }
        return root;
    }

    public boolean search(int val) {
        return searchRecursive(root, val);
    }

    private boolean searchRecursive(TreeNode root, int val) {
        if (root == null) {
            return false;
        }
        if (val == root.val) {
            return true;
        } else if (val < root.val) {
            return searchRecursive(root.left, val);
        } else {
            return searchRecursive(root.right, val);
        }
    }
}
      </pre>
      <p><strong>Key Aspects to Highlight:</strong> Understanding BST properties (left < root < right), recursive implementation for insert and search, handling the base case (empty tree/node), traversing left or right based on comparison, time complexity (O(h) where h is the height of the BST, O(log n) for a balanced BST, O(n) for a skewed BST).</p>
    `
  },
  {
    "question": "Write a function to find the 'k'th largest element in an unsorted array.",
    "description": "Similar to finding the 'k'th smallest, but for the largest element.",
    "hint": "You can sort the array in descending order and return the element at index k-1, or adapt selection algorithms.",
    "answer": `
      <p><strong>Sample Answer (Python - Using Sorting):</strong></p>
      <pre><code class="language-python">def kth_largest(arr, k):
    arr.sort(reverse=True)
    return arr[k - 1]
      </code></pre>
      <p><strong>Sample Answer (Java - Using Sorting):</strong></p>
      <pre><code class="language-java">import java.util.Arrays;

public class KthLargest {
    public static int kthLargest(int[] arr, int k) {
        Arrays.sort(arr);
        return arr[arr.length - k];
    }
}
      </pre>
      <p><strong>Key Aspects to Highlight:</strong> Understanding the requirement, using sorting (mentioning potential for more efficient algorithms like Quickselect), sorting in descending order (or accessing from the end after ascending sort), time complexity (O(n log n) due to sorting).</p>
    `
  },

  {
    "question": "Write a function to implement a basic queue using only two arrays.",
    "description": "This question tests your understanding of queue behavior and how to simulate it with array limitations.",
    "hint": "Use one array for enqueue operations and another for dequeue operations. When the dequeue array is empty, transfer elements from the enqueue array in reverse order.",
    "answer": `
      <p><strong>Sample Answer (Python):</strong></p>
      <pre><code class="language-python">class QueueUsingArrays:
    def __init__(self):
        self.enqueue_arr = []
        self.dequeue_arr = []

    def enqueue(self, item):
        self.enqueue_arr.append(item)

    def dequeue(self):
        if not self.dequeue_arr:
            while self.enqueue_arr:
                self.dequeue_arr.append(self.enqueue_arr.pop())
        if self.dequeue_arr:
            return self.dequeue_arr.pop()
        return None

    def is_empty(self):
        return not self.enqueue_arr and not self.dequeue_arr

    def size(self):
        return len(self.enqueue_arr) + len(self.dequeue_arr)
      </code></pre>
      <p><strong>Sample Answer (Java):</strong></p>
      <pre><code class="language-java">import java.util.ArrayList;
import java.util.List;
import java.util.Collections;

public class QueueUsingArrays<T> {
    private List<T> enqueueArr = new ArrayList<>();
    private List<T> dequeueArr = new ArrayList<>();

    public void enqueue(T item) {
        enqueueArr.add(item);
    }

    public T dequeue() {
        if (dequeueArr.isEmpty()) {
            Collections.reverse(enqueueArr);
            dequeueArr.addAll(enqueueArr);
            enqueueArr.clear();
        }
        if (!dequeueArr.isEmpty()) {
            return dequeueArr.remove(dequeueArr.size() - 1);
        }
        return null;
    }

    public boolean isEmpty() {
        return enqueueArr.isEmpty() && dequeueArr.isEmpty();
    }

    public int size() {
        return enqueueArr.size() + dequeueArr.size();
    }
}
      </code></pre>
      <p><strong>Key Aspects to Highlight:</strong> Using two arrays, enqueue operation (add to the first array), dequeue operation (pop from the second array, transferring and reversing from the first if the second is empty), time complexity (enqueue O(1), dequeue O(n) in the worst case when transferring, O(1) amortized), space complexity (O(n) to store the queue).</p>
    `
  },
  {
    "question": "Write a function to check if two given binary trees are structurally identical (same shape and node values).",
    "description": "This question tests your understanding of binary tree structure and recursive comparison.",
    "hint": "Recursively compare the roots and their corresponding left and right subtrees. Two trees are identical if their roots have the same value and their left and right subtrees are also identical.",
    "answer": `
      <p><strong>Sample Answer (Python):</strong></p>
      <pre><code class="language-python">class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def are_identical_trees(root1, root2):
    if not root1 and not root2:
        return True
    if not root1 or not root2 or root1.val != root2.val:
        return False
    return (are_identical_trees(root1.left, root2.left) and
            are_identical_trees(root1.right, root2.right))
      </code></pre>
      <p><strong>Sample Answer (Java):</strong></p>
      <pre><code class="language-java">public class IdenticalTrees {
    public static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode() {}
        TreeNode(int val) { this.val = val; }
        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }

    public static boolean areIdenticalTrees(TreeNode root1, TreeNode root2) {
        if (root1 == null && root2 == null) {
            return true;
        }
        if (root1 == null || root2 == null || root1.val != root2.val) {
            return false;
        }
        return areIdenticalTrees(root1.left, root2.left) && areIdenticalTrees(root1.right, root2.right);
    }
}
      </pre>
      <p><strong>Key Aspects to Highlight:</strong> Recursive approach, base cases (both nodes are null, one is null and the other isn't, values are different), recursively comparing the left and right subtrees, time complexity (O(n) where n is the number of nodes in the smaller tree in the worst case).</p>
    `
  },
  {
    "question": "Write a function to find the maximum depth of a binary tree using iteration.",
    "description": "This is a variation of the tree depth question, specifically asking for an iterative solution.",
    "hint": "Use a queue to perform a level order traversal. Keep track of the number of levels visited.",
    "answer": `
      <p><strong>Sample Answer (Python):</strong></p>
      <pre><code class="language-python">from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def max_depth_iterative(root):
    if not root:
        return 0
    queue = deque([root])
    depth = 0
    while queue:
        depth += 1
        level_size = len(queue)
        for _ in range(level_size):
            node = queue.popleft()
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    return depth
      </code></pre>
      <p><strong>Sample Answer (Java):</strong></p>
      <pre><code class="language-java">import java.util.LinkedList;
import java.util.Queue;

public class MaxDepthIterative {
    public static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode() {}
        TreeNode(int val) { this.val = val; }
        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }

    public static int maxDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        int depth = 0;
        while (!queue.isEmpty()) {
            depth++;
            int levelSize = queue.size();
            for (int i = 0; i < levelSize; i++) {
                TreeNode node = queue.poll();
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
            }
        }
        return depth;
    }
}
      </pre>
      <p><strong>Key Aspects to Highlight:</strong> Iterative approach using a queue for level order traversal, keeping track of the current depth, processing nodes level by level, time complexity (O(n)), space complexity (O(w) where w is the maximum width of the tree).</p>
    `
  },
  {
    "question": "Why did you choose Infosys as your employer?",
    "description": "Evaluates your motivation and knowledge about Infosys.",
    "hint": "Talk about Infosys’ culture, global presence, and learning opportunities.",
    "answer": `
      <p><strong>Sample Answer:</strong></p>
      <p>I chose Infosys because of its reputation as a global leader in IT services and innovation. The company’s commitment to continuous learning and employee development aligns with my goal to grow technically and professionally. I also appreciate Infosys’ focus on ethical business practices and diversity.</p>
    `
  },
  {
    "question": "How do you stay motivated during repetitive or challenging tasks?",
    "description": "Assesses your work ethic and attitude towards difficult tasks.",
    "hint": "Share your strategies to maintain focus and enthusiasm.",
    "answer": `
      <p><strong>Sample Answer:</strong></p>
      <p>I stay motivated by setting small goals and reminding myself of the bigger picture — how these tasks contribute to my learning and the project’s success. I also take short breaks to refresh my mind and try to find ways to improve or automate repetitive tasks.</p>
    `
  },
  {
    "question": "Tell me about a time when you demonstrated leadership skills.",
    "description": "Evaluates your ability to lead and influence others.",
    "hint": "Use the STAR method to narrate a specific incident.",
    "answer": `
      <p><strong>Sample Answer:</strong></p>
      <p>During a college group project, our team was struggling with task coordination. I volunteered to create a shared timeline and delegated tasks based on strengths. This improved communication and helped us complete the project ahead of schedule, showcasing my leadership and organizational skills.</p>
    `
  },
  {
    "question": "How do you handle feedback and criticism?",
    "description": "Checks your openness to learning and adaptability.",
    "hint": "Give an example of constructive feedback and your positive response.",
    "answer": `
      <p><strong>Sample Answer:</strong></p>
      <p>In my internship, my mentor pointed out that my code lacked proper comments. I took this feedback positively and revised my code with detailed comments. This not only improved the code quality but also helped my team understand it better, demonstrating my willingness to learn and improve.</p>
    `
  },
  {
    "question": "Describe a situation where you had to work with a difficult team member.",
    "description": "Tests your interpersonal and conflict-resolution skills.",
    "hint": "Explain your approach to resolving misunderstandings and collaborating effectively.",
    "answer": `
      <p><strong>Sample Answer:</strong></p>
      <p>In a group assignment, a teammate was uncooperative and missed deadlines. I spoke to them privately to understand their challenges and offered help. By addressing the issue calmly and offering support, we improved collaboration and successfully completed the project.</p>
    `
  },
  {
    "question": "What are your long-term career goals and how does Infosys fit into them?",
    "description": "Assesses your career planning and company fit.",
    "hint": "Show alignment between your aspirations and Infosys’ opportunities.",
    "answer": `
      <p><strong>Sample Answer:</strong></p>
      <p>My long-term goal is to become a skilled software engineer with expertise in emerging technologies like AI and cloud computing. Infosys’ training programs and diverse projects provide the perfect platform for me to develop these skills and grow within a global IT leader.</p>
    `
  }
    ];

    fasalQuestions.forEach((question, index) => {
        const questionDiv = document.createElement('div');
        questionDiv.classList.add('question-item');

        const title = document.createElement('h3');
        title.textContent = `${index + 1}. ${question.question}`;

        const description = document.createElement('p');
        description.textContent = question.description;

        // Create button container
        const buttonContainer = document.createElement('div');
        buttonContainer.style.display = 'flex';
        buttonContainer.style.gap = '10px';
        buttonContainer.style.marginTop = '15px';

        // Hint Button
        const hintButton = document.createElement('button');
        hintButton.textContent = 'Show Hint';
        hintButton.style.padding = '10px 20px';
        hintButton.style.border = 'none';
        hintButton.style.borderRadius = '5px';
        hintButton.style.backgroundColor = '#4CAF50';
        hintButton.style.color = 'white';
        hintButton.style.fontWeight = 'bold';
        hintButton.style.cursor = 'pointer';
        hintButton.style.transition = 'all 0.3s ease';
        hintButton.style.boxShadow = '0 2px 5px rgba(0,0,0,0.2)';
        
        // Hover effect for hint button
        hintButton.addEventListener('mouseover', () => {
            hintButton.style.backgroundColor = '#45a049';
            hintButton.style.transform = 'translateY(-2px)';
            hintButton.style.boxShadow = '0 4px 8px rgba(0,0,0,0.2)';
        });
        
        hintButton.addEventListener('mouseout', () => {
            hintButton.style.backgroundColor = '#4CAF50';
            hintButton.style.transform = 'translateY(0)';
            hintButton.style.boxShadow = '0 2px 5px rgba(0,0,0,0.2)';
        });
        
        hintButton.addEventListener('click', () => {
            hintQuestionTitle.textContent = question.question;
            hintContentDiv.innerHTML = `<p>${question.hint}</p>`;
            hintModal.style.display = 'block';
        });

        // Answer Button
        const answerButton = document.createElement('button');
        answerButton.textContent = 'Show Answer';
        answerButton.style.padding = '10px 20px';
        answerButton.style.border = 'none';
        answerButton.style.borderRadius = '5px';
        answerButton.style.backgroundColor = '#2196F3';
        answerButton.style.color = 'white';
        answerButton.style.fontWeight = 'bold';
        answerButton.style.cursor = 'pointer';
        answerButton.style.transition = 'all 0.3s ease';
        answerButton.style.boxShadow = '0 2px 5px rgba(0,0,0,0.2)';
        
        // Hover effect for answer button
        answerButton.addEventListener('mouseover', () => {
            answerButton.style.backgroundColor = '#0b7dda';
            answerButton.style.transform = 'translateY(-2px)';
            answerButton.style.boxShadow = '0 4px 8px rgba(0,0,0,0.2)';
        });
        
        answerButton.addEventListener('mouseout', () => {
            answerButton.style.backgroundColor = '#2196F3';
            answerButton.style.transform = 'translateY(0)';
            answerButton.style.boxShadow = '0 2px 5px rgba(0,0,0,0.2)';
        });
        
        answerButton.addEventListener('click', () => {
            hintQuestionTitle.textContent = question.question;
            hintContentDiv.innerHTML = question.answer;
            hintModal.style.display = 'block';
        });

        // Add buttons to container
        buttonContainer.appendChild(hintButton);
        buttonContainer.appendChild(answerButton);

        questionDiv.appendChild(title);
        questionDiv.appendChild(description);
        questionDiv.appendChild(buttonContainer);
        questionsListDiv.appendChild(questionDiv);
    });

    closeBtn.addEventListener('click', () => {
        hintModal.style.display = 'none';
    });

    window.addEventListener('click', (event) => {
        if (event.target === hintModal) {
            hintModal.style.display = 'none';
        }
    });
});