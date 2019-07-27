---
layout: post
title:  "Hadoop HDFS"
date:   2016-03-07
categories: BigData
excerpt: 
tags: Hadoop BigData HDFS
---

* content
{:toc}

## What's HDFS
HDFS, the Hadoop Distributed File System, a distributed file system with high-throughput. Files are stored in a redundant fashion across multiple machines to ensure their durability to failure and high availability to very parallel applications. 


## Distributed File System Basics

### NFS, the Network File System
NFS provides remote access to a single logical volume stored on a single machine. An NFS server makes a portion of its local file system visible to external clients.

缺点： 

* 数据集中存放，一旦server down了后果很严重。
* 如果在某一时刻访问量太大，一台机子处理能力有限
* 所有的人都要把数据拷到自己的本地处理

## Hadoop Distributed File System

Hadoop的几个特征能避免NFS的flaw:

* 将大文件分割成block-based的小块，存放在多个cluster上
* 文件有多个copy，可以避免一个机器出错导致数据丢失，更可靠 (3 copy by default)

![hdfs](/images/hdfs.png)

The NameNode stores all the metadata for the file system. Metadata 通常较小, all of this information can be stored in the main memory of the NameNode machine, allowing fast access to the metadata.

To open a file, client 先从NameNode得到应该去哪些dataNode取对应的data block，然后去对应的dataNode进行数据读取。在这里nameNode并不参与实际数据的传输工作，它像一个调控器。NameNode fail的后果比dataNode fail的后果严重，当DataNode fail，程序可以照常进行，当NameNode fail，必须manually去恢复(虽然我们有冗余meta data的备份)。但是通常情况下，NameNode做的工作都是比较light的工作，fail的可能性也没有那么大。


## Reference
https://developer.yahoo.com/hadoop/tutorial/module2.html