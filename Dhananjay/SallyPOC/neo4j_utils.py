import re
# Define a function to delete all nodes and relationships
def delete_all_nodes_and_relationships(driver):
    with driver.session() as session:
        # Delete all nodes and relationships
        delete_query = """
        MATCH (n)
        OPTIONAL MATCH (n)-[r]-()
        DELETE n, r
        """
        session.run(delete_query)

        print("All nodes and relationships deleted successfully.")


# Define a function to create a node with an array property
def add_triplet_to_neo4j(driver,subject,subject_tag, subject_sali_tags, predicate, object,object_tag, object_sali_tags):
    subject_sali_tags.append(subject_tag)
    object_sali_tags.append(object_tag)
    with driver.session() as session:
        #Create a node with an array property
        create_node_query = f"CREATE (:{subject} {{tags: $subject_sali_tags_1}})-[:{predicate}]->(:{object} {{tags: $object_sali_tags_1}})"
        session.run(create_node_query, subject_sali_tags_1=subject_sali_tags, object_sali_tags_1=object_sali_tags)
    

        print("Triplet added successfully. Subject: "+subject+" Predicate: "+predicate+" Object: "+object)

def create_node(driver,object,predicate,subject):
    object=re.sub(r'\W+', '_',object)
    predicate=re.sub(r'\W+', '_',predicate)
    subject=re.sub(r'\W+', '_',subject)
    with driver.session() as session:
        #Create a node with an array property
        create_node_query = f"CREATE (:{object})-[:{predicate}]->(:{subject})"
        session.run(create_node_query, subject=subject)
    

        print("Triplet added successfully. Subject: "+subject+" Predicate: "+predicate+" Object: "+object) 

def merge_nodes_by_name(driver,label):
    label=re.sub(r'\W+', '_',label)
    with driver.session() as session:
        #merge_query = f" MATCH (n:{label}) WITH n.name AS name, COLLECT(n) AS nodelist, COUNT(*) AS count WHERE count > 1 CALL apoc.refactor.mergeNodes(nodelist) YIELD node "

        merge_query = f"""
                    MATCH (n:{label})
                    WITH n.name AS name, COLLECT(n) AS nodelist, COUNT(*) AS count
                    WHERE count > 1
                    CALL apoc.refactor.mergeNodes(nodelist) YIELD node
                    RETURN node
                    """

        session.run(merge_query, label=label)
        
        print("Nodes merged successfully. Label: "+label)