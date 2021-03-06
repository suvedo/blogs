[*<<返回主页*](../index.md)<br><br>
**本文为作者原创，转载请注明出处**<br>
### 字符串匹配的三种场景及其解法
本文讲解字符串匹配相关算法。<br><br>
字符串匹配就是给定两个字符串集合A和B，求A和B是否匹配。具体来说，如果集合A、B都只包含一个字符串（即字符串A和字符串B），则是一对一匹配场景，要解决的问题是字符串B是否是字符串A的子串（与子序列区别）；如果集合A包含多个字符串，集合B只包含一个字符串（即字符串B），则是一对多匹配场景，要解决的问题是字符串B是否出现在集合A中（是否是A的元素）；如果集合A和集合B都包含多个字符串，则是多对多匹配场景，要解决的问题是集合A、B的交集有多少。<br><br>
#### 一对一匹配
一对一匹配经典的解法是KMP算法。KMP算法的核心思想是利用字符串B本身的结构关系来减少计算量，使得一旦遇到与A中某个字符不匹配时，可以不从头匹配。所以首先要计算字符串B的最大前缀后缀匹配（保存为next数组），然后再去快速匹配字符串A。<br><br>
首先计算字符串B的next数组。next数组的长度与B的长度相同，且均从0开始索引，next\[i\]的值表示子串B\[0,1,...,i\]的后缀与其前缀的最大匹配值（即子串B\[0,1,...,i\]最多有多少前缀和其后缀相同）。
计算时，利用动态规划的思想，假设next\[i-1\]已知，即子串B\[0,1,...,i-1\]的最大匹配后缀前缀已知，为next\[i-1\]个，则：如果B\[i\] = B\[next\[i-1\]\]，则next\[i\] = next\[i-1\]+1，如果B\[i\] != B\[next\[i-1\]\]，此时next\[i\]必定小于next\[i-1\]+1（即子串B\[0,1,...i\]的最大匹配前缀后缀可能有next\[i-1\],next\[i-1\]-1,next\[i-1\]-2,...,0个），
首先检查B\[0,1,...next\[i-1\]-1\]和B\[i-next\[i-1\]+1,...,i\]是否相等，由于B\[0,1,....next\[i-1\]-1\]与B\[i-next\[i-1\],...,i-1\]相等，其实就是去找子串B\[i-next\[i-1\],...,i\]的最大匹配后缀前缀，由于B\[i-next\[i-1\],...,i-1\]等于B\[0,...,next\[i-1\]-1\]，且B\[0,...,next\[i-1\]-1\]的最大匹配前缀后缀已知，为next\[next\[i-1\]-1\]，所以子串B\[i-next\[i-1\],...,i-1\]的最大匹配前缀后缀也为next\[next\[i-1\]-1\]，因此只需检查B\[next\[next\[i-1\]-1\]\]是否等于B\[i\]，如果相等，则next\[i\] = next\[next\[i-1\]-1\]+1，如果不等，则继续往前找，知道找到B\[0\]处。<br><br>
求出next数组后，一旦遇到不能匹配的位置i，由于i之前的文本都匹配上了说明B\[0,...,i-1\] =
A\[s,...,s+i-1\]，又由next\[i-1\]知道，B\[0,...,i-1\]的前缀和A\[s,...,s+i-1\]的后缀最大匹配为next\[i-1\]，此时可以从B字符串的next\[i\]处开始匹配，而A继续从(s+i-1)处匹配，不再是每次遇到不匹配的时候只能往后移动一位从头匹配，大大减小了计算量。<br><br>
#### 一对多匹配
一对多匹配经典的解法是：字典树，按前缀查询。详见[字典树的五种实现方式](../data_structure/2_trie_tree.md)<br><br>
#### 多对多匹配
多对多匹配的解法综合了一对一匹配和一对多匹配的解法，即将其中一个集合B构造成一种特殊的字典树（该字典树包含了类似KMP算法中的next数组的信息），在集合A上查询该特殊字典树中的所有字符串，这种特殊的字典树就是AC自动机。<br><br>
AC自动机本质是字典树+KMP。相比KMP匹配算法，运行一次AC自动机可以匹配字典B里的所有目标串（而不是像KMP算法一次只匹配一个字符串），其时间复杂度为O(m)（m为字符串长度），如果集合A中有n个元素，则每个元素运行一次AC自动机即可；
之所以一次可以匹配集合B里的所有字符串，就是因为利用了字典构造了字典树，相比字典树匹配算法，只需要运行一次AC自动机就可以匹配到m里的所有位置（而字典树匹配则需要在m的每个位置从根节点进行一次匹配），所以多对多的匹配只需要O(m)的复杂度。<br><br>
接下来看看怎么构造AC自动机（字典树+KMP）：<br><br>
1）首先利用字典集合B里的所有词构造一颗字典树；<br>
2）字典树的每个节点除了孩子节点的指针、是否为根节点外，还包括了fail指针，fail指针是指在当前节点的所有孩子节点中都匹配失败后要从AC自动机的什么位置开始匹配，具体来说fail指针指向根节点到当前节点组成的子串的最大匹配前缀后缀；<br>
&nbsp;&nbsp;&nbsp;&nbsp;2.1）fail指针构建过程：对1）中构建好的字典树进行广度优先遍历，从队列中弹出一个节点cur，节点cur的fail指针已经构建好，现在来构建cur节点的所有孩子节点的fail指针，看cur.fail.child\[i\]是否等于cur.child\[i\]，如果等于，则cur.child\[i\].fail = cur.fail.child\[i\]，否则继续看cur.fail.fail的child\[i\]是否等于cur.child\[i\]，直到fail等于根节点；<br>
&nbsp;&nbsp;&nbsp;&nbsp;2.2）构建fail指针跟KMP算法中构造next数组的思想一样，但是是构造多个字符串的next数组，所以构建fail指针的方法既有有动态规划的思想，又有遍历树的思想（层次遍历）；<br><br>
构建好自动机之后，就可以A中的字符串m进行匹配了，具体过程如下：<br><br>
1）取出m的当前字符（从0开始往后遍历），以及自动机的当前节点（初始时刻为root节点）;<br>
2）若当前字符与当前节点的某个孩子节点child\[i\]匹配，则看孩子节点child\[i\]、孩子节点child\[i\]的fail指针指向的节点child\[i\].fail、child\[i\].fail.fail.....(直到fail为null)是否是某个字符串的结尾，如果是，则匹配到了字典里的词；然后将匹配到的孩子节点赋为当前节点，继续执行2）;<br>
3）若当前字符与当前节点的所有孩子节点都不匹配，则当前节点等于fail指向的节点，继续执行2），如果fail指针为root，则取文本串的下一个字符，相当于重启AC自动机；<br><br>
#### 参考文献
算法导论(第三版):P588~P594<br><br>
[多模字符串匹配算法之AC自动机—原理与实现](https://www.cnblogs.com/nullzx/p/7499397.html)<br>
