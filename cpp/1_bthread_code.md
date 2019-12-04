[*<<返回主页*](../index.md)<br><br>
**本文为作者原创，转载请注明出处**<br>

### bthread源码阅读
bthread是为brpc设计的m:n线程库，其中m表示bthread（用户线程）的数量，n表示pthread（worer线程）的数量，一般m远大于n。通过bthread线程库，brpc可以同时兼顾scalability和cache locality，scalability通过pthread间偷bthread实现，cache locality通过同一个pthread中运行不同的bthread实现。用户通过bthread_start_background()、bthread_start_urgent()等接口即可创建并执行一个线程。<br><br>
#### int bthread_start_background(bthread_t tid, const bthread_attr_t\* attr, void\* (\*fn)(void\*), void\* arg) __THROW
1、函数描述： Create bthread 'fn(args)' with attributes 'attr' and put the identifier into 'tid'<br><br>
2、每个pthread线程有个独立的__thread TaskGroup tls_task_group对象（tls = thread local storage：利用__thread修饰，__thread变量每个线程有一份独立实例），这个对象维护了当前pthread的任务队列等信息；如果该pthread线程未曾创建TaskGroup（即__thread bthread::tls_task_group指针为空，没有任何worker），则进入bthread::start_from_non_worker(tid, attr, fn, arg)，否则调用bthread::tls_task_group->start_background<false>(tid, attr, fn, arg);<br><br>
2.1、bthread::start_from_non_worker(tid, attr, fn, arg)：<br><br>
2.1.1、首先获取TaskControl (TaskControl* c = get_or_new_task_control())，TaskControl管理所有的pthread线程，负责bthread在各pthread之间调度与协调等工作，程序维护一个全局的TaskControl，这个TaskControl所有pthread线程共享，不是thread local的，每个线程访问时将其指针转换为原子变量的指针；如果TaskControl不为空，则直接返回，否则新建一个：对全局TaskControl加锁，新建并初始化TaskControl（调用TaskControl->init(concurrency)，concurrency表示pthread数量，即配置的并发数），初始化TaskControl的步骤：<br><br>
1）初始化global_timer_thread(略过)；<br><br>
2）（非阻塞地）创建并启动concurrency个pthread，并保存在vector中，每个pthread worker有个work_thread()线程函数，该线程函数传入全局的TaskControl指针作为参数；针对每个pthread:<br><br>
a）work_thread()首先调用开始函数g_worker_startfn()，这是一个全局的函数指针；<br><br>
b）然后调用TaskControl的create_group()创建TaskGroup（TaskControl是work_thread()的参数，所以可以调用TaskControl的成员函数，而每个Pthread对应一个work_thread()线程函数，即每个pthread对应一个TaskGroup，因为有个thread local的tls_task_group，所以这两个指向同一个对象，均由TaskControl创建；创建的TaskGroup有个main_tid）<br><br>
c）创建TaskGroup的步骤如下：1)new TaskGroup：传入TaskControl的指针；2)初始化new出来的TaskGroup（TaskGroup->init(FLAGS_task_group_runqueue_capacity)，传入每个TaskGroup的运行队列的最大长度，每个TaskGroup有一个WorkStealingQueue<bthread_t> _rq的运行队列，还有一个RemoteTaskQueue _remote_rq的队列，后者容量是前者的一半，每个TaskGroup有一个TaskMeta类型的_cur_meta作为TaskGroup的元数据，主要有fn,arg,attr,tid,stk等，_main_tid即是_cur_meta中的tid，_main_stk即_cur_meta中的stk；3)TaskControl::_add_group(g)：将步骤1)和2)中创建好的TaskGroup加入TaskGroup** TaskControl::_groups里；4）返回创建的TaskGroup。<br><br>
d）然后调用g->run_main_task()，g即为create_group()创建出的TaskGroup*：1)while(1) { 1)wait_task(&tid)，系统调用，等待条件成立； 2)执行TaskGroup::sched_to(&dummy, tid) }<br><br>
2.1.2、调用TaskGroup* c -> choose_one_group() 获取一个TaskGroup*：随机从现有的_groups中选择一个TaskGroup*返回；<br><br>
2.1.3、调用TaskGroup的start_backgroud<true>(tid, attr, fn, arg)：构造TaskMeta，并把fn, arg赋值给TaskMeta，make_tid()，获取tid，然后调用ready_to_run_remote(),ready_to_run_remote(tid, bool nosignal)做的事情：a)将tid加入_remote_rq中；b)向TaskControl发信号<br><br>
2.2、bthread::tls_task_group->start_background<false>(tid, attr, fn, arg): 与2.1.3中的bthread::tls_task_group->start_background<true>(tid, attr, fn, arg)的区别是：调用ready_to_run()，而不是ready_to_run_remote()，具体来说：1)将tid加入_rq中，而不是_remote_rq中；2）向TaskControl发信号，与remote相同；<br>

#### int bthread_start_urgent(bthread_t* tid, bthread_attr_t* attr, void* (*fn)(void*), void* arg) __THROW
获取thread local的tls_task_group，如果为空，则调用start_from_non_worker()，与bthread_start_background()中完全相同；否则调用g->start_foreground()<br>

#### 关于parking_lot：
1、TaskControl里维护PARKING_LOT_NUM（目前版本定义为4，不能配置）个parking_lot，每个worker线程根据pthread id哈希到某一个parking_lot上<br><br>
2、每个parklot维护一个原子变量，这个原子变量用于futex做同步，哈希到同一个parklot上的线程由于共享parkinglot，也即共同竞争这个原子变量，相互之间wait和wakeup<br><br>
3、某个线程或taskGroup被唤醒之后，首先去remote_queue中pop任务，然后在去别的taskGroup中偷任务<br><br>
4、去别的taskGroup中偷任务的时候，随机取一个taskGroup，首先从这个taskGroup的runqueue中偷，然后从remotequeue中pop；<br><br>
5、singal的时候，先singal自己属于的那个parklot，如果唤醒的线程不够，在singal另外三个parkinglots<br><br>

#### 参考文献
[brpc](https://github.com/apache/incubator-brpc)
