import networkx as nx
import pytest

from pidsmaker.incidents import EdgeRef, Incident


def test_incident_exposes_its_time_bounds_and_edges():
    graph = nx.MultiDiGraph()
    graph.add_edge(10, 20)
    graph.add_edge(20, 30)

    seed = EdgeRef(srcnode=10, dstnode=20, time=200, edge_type=1, score=0.95)
    connector = EdgeRef(srcnode=20, dstnode=30, time=250, edge_type=2, score=0.20)

    incident = Incident(
        graph=graph,
        seed_edges=(seed,),
        connector_edges=(connector,),
    )

    assert incident.edges == (seed, connector)
    assert incident.start_time == 200
    assert incident.end_time == 250


def test_incident_rejects_an_empty_seed_set():
    with pytest.raises(ValueError, match="at least one seed"):
        Incident(graph=nx.MultiDiGraph(), seed_edges=())


def test_incident_rejects_a_seed_outside_its_graph():
    graph = nx.MultiDiGraph()
    graph.add_node(10)
    seed = EdgeRef(srcnode=10, dstnode=99, time=200, edge_type=1, score=0.95)

    with pytest.raises(ValueError, match="belong to the incident graph"):
        Incident(graph=graph, seed_edges=(seed,))
