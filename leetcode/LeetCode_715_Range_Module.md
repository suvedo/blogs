[*<<返回主页*](../index.md)
### [LeetCode 715. Range Module](https://leetcode.com/problems/range-module/description/)

题目要求高效地实现区间的添加，删除和查询操作。

题目分析：本题是一道设计题，要求实现一个类(Range Module)的三个方法：addRange(int, int), queryRange(int, int) 和 removeRange(int, int)。
  该类的对象会穿插地调用这三个方法。听起来有点像并查集，但并查集“一般”不涉及删除操作（不是很确定，只能说“一般”），这道题其实用简单的map就可以实现。
  具体来说：map的key是区间的起点，value是区间的终点（其实用set也可以实现，set的元素是个pair<int, int>,自己实现比较函数即可）。
  由于map的key是有序的，所以在添加，删除或者查询区间的时候可以用log(n)的时间快速定位到用执行区间操作的位置。
  
  在每次进行区间操作（add, remove or query）之前，区间与区间之间都没有重叠部分，且区间与区间不相邻。
  
  addRange(int left, int right)：如果对象里没有任何区间，则直接插入要加的区间即可。否则，利用map<int, int>的lower_bound(int)函数查找第一个左区间大于或者等于left位置iter（时间复杂度是log(n)），接下来分三种情况：
1. 如果iter == map::end，说明所有左区间都小于left（包括最后一个区间）,此时a)如果最后一个区间的右区间小于left，直接插入```[left, right)```到末尾即可；b)否则，将最后一个区间的右区间只改为右区间和right中较大的一个即可。
2. 如果iter == map::begin,说明在iter之前没有任何区间，此时从iter开始迭代区间，直到iter == map::end 或者iter的左区间大于left。迭代过程中如果发现当前区间与```[left, right)```没有交集且不相邻，则跳过此区间，如果相交或者相邻，则将left赋为left和当前左区间中较小的一个，right赋为right和当前右区间较大的一个，并删除当前区间。迭代结束之后将```[left, right)```加入到map中。
3. iter在除以上两种情况外的位置，此时iter前的一个区间可能与```[left, right)```也有交集，所以先将iter--，然后在执行步骤2的操作即可。
 
removeRange(int left, int right):如果对象里没有任何区间，直接返回。 否则，利用map<int, int>的lower_bound(int)函数查找第一个左区间大于或者等于left位置iter（时间复杂度是log(n)），接下来分三种情况：
1. 如果iter == map::end，说明所有左区间都小于left（包括最后一个区间）,此时a)如果最后一个区间的右区间小于等于left，直接返回即可；b)如果最后一个区间的右区间小于等于right,只需将最后一个右区间的改为left即可；c)如果最后一个区间的右区间小于right，除了将最后一个右区间的改为left，还有插入```[right, 最后一个区间的右区间)```到对象中。
2. 如果iter == map::begin,说明在iter之前没有任何区间，此时从iter开始迭代区间，直到iter == map::end 或者 iter的左区间大于等于left。迭代过程中如果发现当前区间与```[left, right)```没有交集，则跳过此区间，如果相交，可分为以下四种情况:（当前区间的左区间用first表示，当前区间的右区间用second表示）</br>
a) 如果first <= left && second > right，插入```[second, right)```当对象中，并删除当前区间；</br>
b) 如果first <= left && second <= right，删除当前区间即可；</br>
c) 如果first > left && second <= right，将当前区间的右区间改为left即可；</br>
d) 如果first > left && second > right，将```[right, second)```插入对象中，并将当前区间的右区间值改为left。</br>
3. iter在除以上两种情况外的位置，此时iter前的一个区间可能与```[left, right)```也有交集，所以先将iter--，然后在执行步骤2的操作即可。
    
queryRange(int left, int right)：如果对象里没有任何区间，返回false。否则，利用map<int, int>的lower_bound(int)函数查找第一个左区间大于或者等于left位置iter（时间复杂度是log(n)），接下来分三种情况：</br>
1. 如果iter == map::end，说明所有左区间都小于left（包括最后一个区间），此时只需考察最后一个区间的右区间是否大于等于right，是则返回true，否则返回false;
2. 如果iter == map::begin,情况与1相同，只需考察第一个区间的右区间是否大于等于right，是则返回true，否则返回false;
3. iter在除以上两种情况外的位置，此时iter前的一个区间可能与```[left, right)```也有交集，除了考察当前区间是否包含```[left, right)```以外， 还有考察iter的前一个区间是否包含```[left, right)```。
 
```
    class RangeModule {
      map<int, int> range;
    public:
      RangeModule() {

      }
    
      void addRange(int left, int right) {
          if(range.empty()) {
              range.insert(make_pair(left, right));
              return;
          }
          auto iter = range.lower_bound(left);
          if(iter == range.end()) {
              auto end = range.rbegin();
              int a = end -> first, b = end -> second;
              if(b < left) {
                  range.insert(make_pair(left, right));
              } else {
                  range[a] = max(right, b);
              }
          } else {
              if(iter != range.begin()) iter --;
              while(iter != range.end() && iter -> first <= right) {
                  if(iter -> second < left) {
                      iter ++;
                      continue;
                  }
                  left = min(left, iter -> first);
                  right = max(right, iter -> second);
                  iter = range.erase(iter);
              }
              range.insert(make_pair(left, right));
          }
      }
    
      bool queryRange(int left, int right) {
          if(range.empty()) return false;
          auto iter = range.lower_bound(left);
          if(iter == range.end()) return right <= range.rbegin() -> second;
          if(left >= iter -> first && right <= iter -> second) return true;
          if(iter == range.begin()) return false;
          iter --;
          return left >= iter -> first && right <= iter -> second;
      }
    
      void removeRange(int left, int right) {
          if(range.empty()) return;
          auto iter = range.lower_bound(left);
          if(iter == range.end()) {
              auto end = range.rbegin();
              int a = end -> first, b = end -> second;
              if(b <= left) return;
              if(end -> second > right) {
                  range[right] = b;
              }
              range[a] = left;
          } else {
              if(iter != range.begin()) iter --;
              while(iter != range.end() && iter -> first < right) {
                  if(iter -> second <= left) {
                      iter ++;
                      continue;
                  }
                  if(iter -> first >= left) {
                      if(iter -> second > right) {
                          range.insert(make_pair(right, iter -> second));
                      }
                      iter = range.erase(iter);
                  } else {
                      if(iter -> second > right) {
                          range.insert(make_pair(right, iter -> second));
                      }
                      range[iter -> first] = left;
                      iter ++;
                  }
              }
          }
      }
};
```

