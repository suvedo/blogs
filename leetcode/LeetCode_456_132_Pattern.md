[*<<返回主页*](../index.md)
### [LeetCode. 456 132 Pattern] https://leetcode.com/problems/132-pattern/description/

### 题目描述：
找出整型数组nums是否存在“132”模式，（“132”模式即，存在i < j < k，满足nums[i] < nums[k] < nums[j]），
如果存在返回true，否则返回false。</br>

### 题目分析：

_方法一：Binary Search_</br>
要确定三个下标i, j, k，暴力求解则需要O(n^3)的时间复杂度，而本题并不需要返回具体哪些有哪些"132"pattern，
甚至不需要返回所有的"132"pattern的个数，只需要返回是否存在"132"pattern，因此暴力求解的办法显得太多余。
基于此，可以暴力搜索其中一个值（i.e. 下标i）,而其它两个值或其中一个值采用诸如空间换时间，滑动窗口，
二分查找等技巧降低时间复杂度，时间复杂度最低可降至O(n)（暴力搜索其中一个值得复杂度）。</br>
基于上述分析，可以利用一个数组保存截止到当前下标的最小值，此最小值作为"132"pattern中的最小值，而"132" patern
中的最大值可以用暴力搜索的方式,而次大值采用二分查找的方式。具体步骤如下：从末尾搜索"132"pattern中的最大值，并用
set保存所有搜索过的值，即set中保存的是当前下标后面的所有值，利用二分搜索查找set中是否存在大于mini\[i\]小于nums\[i\]的数，
如果存在，返回true，否则继续搜索。

'''
class Solution {
public:
    bool find132pattern(vector<int>& nums) {
        int n = nums.size();
        if(n < 3) return false;
        vector<int> mini(n, INT_MAX);
        for(int i = 1; i < n; i ++) {
            mini[i] = min(mini[i - 1], nums[i - 1]);
        }
        set<int> st({nums.back()});
        for(int i = n - 2; i > 0; i --) {
            if(nums[i] <= mini[i]) {
                st.insert(nums[i]);
                continue;
            }
            auto iter1 = st.upper_bound(mini[i]), iter2 = st.lower_bound(nums[i]);
            if(iter1 != iter2) return true;
            st.insert(nums[i]);
        }
        return false;
    }
};
'''
时间复杂度：O(nlogn)，空间复杂度：O(n)</br></br>

_方法二：stack_</br>
除了方法一中的技巧之外，还可以用stack替换binary search。具体步骤，从末尾开始搜索132pattern中的较小值，搜索过的值维持一个递减栈，如果当前值小于等于栈顶
元素，直接压栈即可，否则循环弹栈，直到栈顶元素大于mini[i]，弹出的元素不再入栈，（因为弹出的元素都是小于等于mini[i]，而mini[i]是截止到当前的最小值，i以前的
最小值肯定小于等于mini[i]，而弹出的元素小于等于mini[i]，也就是小于等于i之前的最小值，因此弹出的元素肯定不能和i之前的元素构成132pattern，弹出之后不影响
最后结果），如果栈顶元素小于当前元素mini[i]，说明找到了132pattern，返回true，否则当前元素压栈，进入下一次迭代。
'''
class Solution {
public:
    bool find132pattern(vector<int>& nums) {
        int n = nums.size();
        if(n < 3) return false;
        vector<int> mini(n, INT_MAX);
        for(int i = 1; i < n; i ++) {
            mini[i] = min(mini[i - 1], nums[i - 1]);
        }
        stack<int> st;
        for(int i = n - 1; i > 0; i --) {
            if(nums[i] <= mini[i]) continue;
            if(st.empty() || nums[i] <= st.top()) {
                st.push(nums[i]);
            } else {
                while(!st.empty() && st.top() <= mini[i]) {
                    st.pop();
                }
                if(!st.empty() && st.top() < nums[i]) return true;
                st.push(nums[i]);
            }
        }
        return false;
    }
};
'''
时间复杂度：O(n)（因为每个元素顶多入栈一次，出栈一次，栈操作并不影响时间复杂度的上界），空间复杂度：O(n)

