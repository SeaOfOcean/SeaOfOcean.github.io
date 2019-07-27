---
layout: post
title:  "Consistent Hashing"
categories: BigData
excerpt: 
tags: MapReduce BigData
---

* content
{:toc}

## Motivation

Cache can help us get frequetly request data fast. If we have a shared cache for a group of users, e.g. students from some university, the the size of data need to be cached is very large so we may need multiple nodes to store these data.

A naive way is to use a hash function to distribute the data to different machines, by simply adopting k%N, where k is the key of data and N is the number of nodes.

But the problem comes when we want to add server number, most data need to be moved around because the hash value changes.

To solve this problem, consistent hashing is proposed and it is widely adopted in many great systems such as [Memcached](http://memcached.org/) and [Amazon's Dynamo](http://www.allthingsdistributed.com/2007/10/amazons_dynamo.html).

## Consistent Hashing

### Goal

1. How the data is assigned should be independent of the number of servers.
2. Load balance

### Design

The main idea is that in addition to mapping the keys, the server numbers are mapped to the same range.

This approach can be visualized on a circle. 
Servers and objects both hash to points on this circle; an object is stored on the server that is closest in the clockwise direction. If we want to add a server, say server 4, we need to first map server 4 into the circle, and we only need to move the part of keys(between 4 and 2) from server 3 to server 4. So that only this part of data will be influenced.

<center><img src="/images/posts/consistent-hashing.png" width="400"></center>

### Look up and insert

Given a key k, this problem is to find the server s that minimize h(s) subject to h(s)>=h(k).
One good idea is to use a balanced binary search tree, such as a Red-Black tree, to organize the servers and its corresponding hash values.


### Vitual copies

When the server number is small or the load of servers differ, the partition may not be balanced.

The solution to this problem is to add vitual copies of servers. If one server is twice as big as another, it should have twice as many virtual copies.


<center><img src="/images/posts/map.JPG" width="400"><br>Figure from this <a href="http://blog.csdn.net/sparkliang/article/details/5279393">blog</a></center>


## Reference

* [Consistent Hashing and Random Trees](https://www.google.com.sg/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&ved=0ahUKEwio3uiGxd3MAhWIqI8KHQGyC2IQFggaMAA&url=https%3A%2F%2Fwww.akamai.com%2Fes%2Fes%2Fmultimedia%2Fdocuments%2Ftechnical-publication%2Fconsistent-hashing-and-random-trees-distributed-caching-protocols-for-relieving-hot-spots-on-the-world-wide-web-technical-publication.pdf&usg=AFQjCNF6wU7zHgxHj0XINI88h33s2Uw-KA&sig2=1IVbbGQ1e0B00mXW4z757w&bvm=bv.122129774,d.c2I)
* [http://theory.stanford.edu/~tim/s15/l/l1.pdf](http://theory.stanford.edu/~tim/s15/l/l1.pdf)
* [MOOC-Data Manipulation at Scale](https://www.coursera.org/learn/data-manipulation/)
* [The Simple Magic of Consistent Hashing](http://www.paperplanes.de/2011/12/9/the-magic-of-consistent-hashing.html)
* [Consistent hashing](http://michaelnielsen.org/blog/consistent-hashing/)
* [一致性 hash 算法（ consistent hashing ）](http://blog.csdn.net/sparkliang/article/details/5279393)

