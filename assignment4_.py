#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 20:54:42 2017

@author: l3th
"""
from IPython.display import HTML
import folium
import json
import networkx as nx
from folium import plugins
import networkx.classes.function as ncf


from osm import read_way_network, gc_dist
bryggen = read_way_network("bryggen.osm")


def find_nearest_node(graph, lat, lon):
    abs_dist = {}
    for node in graph.__iter__():
        abs_dist[node] = gc_dist(lat, lon, graph.nodes[node]['lat'], graph.nodes[node]['lon'])
    return min(abs_dist, key=abs_dist.get)


start_loc = (55.663811, 12.596087)
goal_loc = (55.665372, 12.578170)

start_node = find_nearest_node(bryggen, 55.663811, 12.596087)
goal_node = find_nearest_node(bryggen, 55.665372, 12.578170)
print(start_node, goal_node)

path = (start_node,)
solutions = []
discovered = set()




def depthFirstSearch(boardList, goalState):
    frontier = set()
    frontierOrdered = [(boardString, "")]
    frontier.add(frontierOrdered[0][0])
    explored = set()
    
    while not len(frontier) == 0:
#        print(frontier, frontierOrdered)
        stateFull = frontierOrdered.pop()
        state = stateFull[0]
        path = stateFull[1]
        explored.add(state)

        if state == goalState:
            return 'Succes'

        for i, d in findNeighbors(state, 'dfs'):
            curState = swapNeighbors(state, int(i))
            if pl<len(path+d): pl = len(path+d)
            if curState not in frontier and curState not in explored:
                frontierOrdered.append((curState, path+d))
                frontier.add(curState)
        
    return 'Failure'

