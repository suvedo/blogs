[*<<返回主页*](/index.md)
### [LeetCode 678. Valid Parenthesis String](https://leetcode.com/problems/valid-parenthesis-string/description/)

**题目描述：**输入一个仅包含```'(',')'```和```'*'```的字符串，判断字符串左右括号是否匹配，其中```'*'```可表示左括号，右括号或者为空。

**题目分析：**</br></br>
_方法一：动态规划_</br>
要计算```[i, j]```区间内的左右括号是否匹配，可以计算```[i, k]``` 和```[k + 1, j]```区间的字符串是否同时匹配，
（其中```i <= k < j```）,如果有一个k值能匹配，说明```[i, j]```匹配，否则，还需要看整个```[i, j]```区间是否匹配:
如果```s[j] == '('```或者```s[i] == ')'```，则不能匹配；否则，如果```s[j] == ')'```且```s[i] == '('``` 
或者 ```s[i] == '*'``` 或者 ```s[j] == '*'```，```dp[i][j] == dp[i + 1][j - 1]```.

```
class Solution {
public:
    bool checkValidString(string s) {
        int n = s.size();
        if(n == 0) return true;
        vector<vector<bool>> dp(n + 1, vector<bool>(n + 1, false));
        for(int i = 0; i < n; i ++) {
            if(s[i] == '*') dp[i][i] = true;
            dp[i + 1][i] = true;
        }
        for(int k = 1; k < n; k ++) {
            for(int i = 0; i + k < n; i ++) {
                if(s[i] == ')' || s[i + k] == '(') continue;
                if(s[i + k] == ')' && s[i] == '(' || s[i] == '*' || s[i + k] == '*') {
                    dp[i][i + k] = dp[i + 1][i + k - 1];
                }
                if(dp[i][i + k]) continue;
                for(int j = i; j < i + k; j ++) {
                    if(dp[i][j] && dp[j + 1][i + k]) {
                        dp[i][i + k] = true;
                        break;
                    }
                }
            }
        }
        return dp[0][n - 1];
    }
};
```
时间复杂度:O(n^3),空间复杂度:O(n^2)</br></br>
_方法二：greedy + stack_</br>
如果原字符串中没有```'*'```字符，只有```'('```和```')'```，可以用栈解决，方法是顺序遍历字符串，遇到```'('```就入栈，遇到```')'```就弹栈。本题中存在类似通配符的```'*'```字符，在上述方法上稍作变通即可：运用两个栈st1和st2，分别存放```'('```和```'*'```的索引，即遍历字符串的过程中，遇到```'('```和```'*'```就分别入栈，遇到```')'```时，如果st1不为空，则弹出st1栈顶元素，否则，如果st2不为空，则弹出st2栈顶元素，如果st2为空，说明没有该```')'```不能与前面的字符匹配，返回false。遍历完成后，如果st1不为空，则比较st2栈顶元素和st1栈顶元素的大小，前者小于后者，说明该```'('```后面没有字符与之匹配，返回false，前者大于后者，st1和st2均弹栈，直到st1或者st2为空。</br>
该算法说明如下：遍历字符串的过程中，当遇到```')'```时，优先弹st1中的元素，不管距离当前右括号最近的是左括号还是```'*'```，此步骤即可理解为贪心法，因为如果此后```'*'```被用作空字符，则st2中的元素没起作用，弹出st1中的元素是自然而然的选择；如果此后```'*'```被用作```'('```，此时先弹st1和st2没有影响，与st1和st2中元素匹配的均出现在这两者的后面，只是st1中和st2中弹出的顺序不一样，并不影响匹配的判断；如果此后```'*'```被当做```')'```，说明在```'*'```之前有```'('```，此时只是st1中两个```'('```弹出顺序不一样，并不影响匹配的判断。所以该贪心选择能得到正确的解。
```
class Solution {
public:
    bool checkValidString(string s) {
        stack<int> st1, st2;
        for(int i = 0; i < s.size(); i ++) {
            if(s[i] == '(') {
                st1.push(i);
                continue;
            }
            if(s[i] == '*') {
                st2.push(i);
                continue;
            }
            if(!st1.empty()) {
                st1.pop();
            } else {
                if(!st2.empty()) {
                    st2.pop();
                } else return false;
            }
        }
        while(!st1.empty()) {
            if(st2.empty() || st2.top() < st1.top()) return false;
            st1.pop();
            st2.pop();
        }
        return true;
    }
};
```
时间复杂度：O(n)，空间复杂度O(n)</br></br>
_方法三：greedy_</br>
方法二中用栈同时保存了左括号和星号的索引位置，而如果原字符串中没有```'*'```，则只需保存当前未被匹配的```'('```的数量，有```'*'```时，```'('```的数量不是一个确定的值，而是一个区间。在本方法中，不在考虑```'*'```，而只考虑当前左括号的所有可能的取值，因为所有可能的取值是连续的，所以只需保存最小值lo和最大值hi即可。初始时刻，lo和hi都为0，遍历字符串时，遇到左括号，lo++并且hi++，遇到```'*'```，由于该```'*'```可以作为左括号，空字符和由括号，所以此时最小值（如果不为0）减1（星号作为右括号），最大值加1（星号作为左括号）。字符串遍历完成后，如果最小值等于0，说明可以匹配，返回true，否则返回false。
```
class Solution {
public:
    bool checkValidString(string s) {
        int lo = 0, hi = 0;
        for(char c : s) {
            if(c == '(') {
                lo ++;
                hi ++;
            } else if(c == '*') {
                lo = lo == 0 ? 0 : (lo - 1);
                hi ++;
            } else {
                if(hi == 0) return false;
                lo = lo == 0 ? 0 : (lo - 1);
                hi --;
            }
        }
        return lo == 0;
    }
};
```
时间复杂度：O(n)，空间复杂度O(1)

