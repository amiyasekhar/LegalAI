from rdflib import Graph, Literal, RDF, URIRef,Namespace, BNode
from rdflib.namespace import FOAF , XSD,OWL,SKOS,RDFS

lmss = Namespace("http://lmss.sali.org/")
# Create a Graph
g = Graph()
# Parse in an RDF file hosted on the Internet
g.parse("LMSS.owl")

#Python wraper for classes in ontology
class OWLClass:
    node_label = OWL.Class.n3(namespace_manager=g.namespace_manager).replace(":","_")
    
    def __init__(self, iri, label, defination):
        self.iri = iri
        self.label = label
        self.defination = defination
        self.parents = []
        self.relations = []
        self.label_embedding = []
        self.defination_embedding = []

class Relation:
    def __init__(self, iri, label, to_iri):
        self.iri = iri
        self.label = label
        self.to_iri = to_iri

#FILL CLASS DETAILS 
onto_classes = g.subjects(RDF.type, OWL.Class)
classes = []
for c in onto_classes:
    
    label = g.value(c, RDFS.label)
    if label is None:
        label = c.split("#")[-1]
        
    
    classes.append(
        OWLClass( c,label, g.value(c, SKOS.definition))
        )
    
classes = classes[35:]

#FILL RELATIONSHIPS DETAILS AND PARENTS

for clas in classes:
    
    parents = g.objects(clas.iri, RDFS.subClassOf)
    
    for p in list(parents):
        
        if type(p) == BNode:
                
            what = g.value(p, OWL.onProperty)
            where = g.value(p, OWL.someValuesFrom)

            if what is None or where is None:
                continue

    
            if what.rsplit('/', 1)[0]+'/' == str(lmss):
                label=g.value(what, RDFS.label)
            else :   
                label=what.n3(namespace_manager=g.namespace_manager)
            
            clas.relations.append(
                Relation(what,label,where)
                )
            
        elif type(p) == URIRef:
            clas.parents.append(p)
            clas.relations.append(
                Relation(
                    RDFS.subClassOf,RDFS.subClassOf.n3(namespace_manager=g.namespace_manager),p
                    )
                )

irir_embedding_mapping = {}
# #read sali mapping
# import json
# json_file_path1 = 'sali_embeddings/sali_mapping_empty.json'
# json_file_path = "sali_embeddings/sali_mapping.json"

# # Open the JSON file for reading
# with open(json_file_path, 'r') as json_file:
#     # Load the JSON data from the file
#     lines = json_file.readlines()

# with open(json_file_path1, 'r') as json_file:
#     # Load the JSON data from the file
#     lines += json_file.readlines()
    
# for line in lines:
#     json_obj = json.loads(line)
#     irir_embedding_mapping[json_obj["iri"]] = json_obj

# print(len(irir_embedding_mapping))

with open("sali_embeddings/sali_relationship_mapping.json", 'r') as json_file:
    # Load the JSON data from the file
    lines = json_file.readlines()

from neo4j import GraphDatabase
import json
import traceback
import tqdm
import time

uri = "bolt://64.227.177.188:7687"
username = "neo4j"
password = "data-virtual-nirvana-paul-brave-1820"

def create_nodes(tx, owl_classes):

    

    for i,owl_class in enumerate(owl_classes):
        query = (
            "CREATE (:{} {{iri: $iri, label: $label, definition: $definition, label_embedding: $label_embedding, definition_embedding:$definition_embedding}})"
            .format(OWLClass.node_label)
        )
        label_embedding = irir_embedding_mapping[str(owl_class.iri)]["label_embedding"]
        if "definition_embedding" in irir_embedding_mapping[str(owl_class.iri)].keys():
            definition_embedding =  irir_embedding_mapping[str(owl_class.iri)]["definition_embedding"]
        else:
            definition_embedding = []
        
        
        #print the query which will be executed
        #print(query)
        tx.run(query, iri=owl_class.iri, label=owl_class.label, definition=owl_class.defination, label_embedding = label_embedding, definition_embedding=definition_embedding)





def create_relationships(owl_classes):
    
    i=4928
    
    driver = GraphDatabase.driver(uri, auth=(username, password))
    session =driver.session()  
    
    for owl_class in tqdm.tqdm(owl_classes[i+1:]):#same as db count

        if owl_class.relations == []:
            continue



        for relation in owl_class.relations:

            
            #get the label and defination of rel.to_iri
            
            label = relation.label.replace(":", "_")

            
            try:


                i=i+1
                json_obj = json.loads(lines[i])
                label_embedding =json_obj["embedding"]
                definition_embedding = []

                to_iri_defination = g.value(relation.to_iri, SKOS.definition)

                if to_iri_defination is not None and owl_class.defination is not None:
                    i=i+1
                    json_obj = json.loads(lines[i])
                    definition_embedding =json_obj["embedding"]

                
                query_r = (
                    f"MATCH (a:{OWLClass.node_label} {{iri: '{owl_class.iri}'}}), "
                    f"(b:{OWLClass.node_label} {{iri: '{relation.to_iri}'}}) "
                    f"CREATE (a)-[:{label}  {{ iri: '{relation.iri}' , label_embedding: { label_embedding } , definition_embedding: {definition_embedding}  }} ]->(b)"
                )

                
                session.run(query_r)
                time.sleep(5)

                if i % 100 == 0:
                    session.close()
                    driver.close()
                    print("memory freeeeee")
                    time.sleep(10)
                    driver = GraphDatabase.driver(uri, auth=(username, password))
                    session =driver.session() 

                # Wait for 5 seconds
                #time.sleep(5)

                
                
                    
            except Exception as e:
                print(e)
                traceback.print_exc()


            
                
        
def delete_all_nodes(tx):
    query = "MATCH (n) DETACH DELETE n"
    tx.run(query)

#session.write_transaction(create_nodes, classes)
create_relationships(classes)
#session.write_transaction(delete_all_nodes)

from rdflib import Graph, Literal, RDF, URIRef
from rdflib.namespace import FOAF , XSD,OWL,SKOS,RDFS

# Create a Graph
g = Graph()

# Parse in an RDF file hosted on the Internet
g.parse("main.owl")

q = """
    PREFIX xml: <http://www.w3.org/XML/1998/namespace>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX dc: <http://purl.org/dc/elements/1.1/>
    PREFIX v1: <http://www.loc.gov/mads/rdf/v1#>
    PREFIX lmss: <http://lmss.sali.org/>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    PREFIX lmss1: <http://lmss.sali.org/2506>
    PREFIX schema: <http://schema.org/>
    PREFIX ontology: <http://www.geonames.org/ontology#>
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>


    SELECT ?r ?s ?t ?l
    WHERE {
        ?p rdf:type owl:Class .
        ?p rdfs:label "Judge" .
        ?p rdfs:subClassOf ?r .
        ?r rdf:type owl:Restriction .
        ?r owl:onProperty ?t.
        ?r owl:someValuesFrom ?s.
    }
"""
for r in g.query(q):
    print(r["r"])
    print(URIRef(r["s"]).n3(g.namespace_manager))
    print(URIRef(r["t"]).n3(g.namespace_manager))
    print(r["l"])

# Loop through each triple in the graph (subj, pred, obj)
for subj, pred, obj in g:
    # Check if there is at least one triple in the Graph
    if (subj, pred, obj) not in g:
       raise Exception("It better be!")
    
    
    print(subj, pred, obj)

    

# Print the number of "triples" in the Graph
# print(f"Graph g has {len(g)} statements.")
# print(OWL.Class)