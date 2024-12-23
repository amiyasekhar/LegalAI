#create function
import requests
base_url = 'https://sali.273v.io'

def get_sali_concept_url(text):
    url = base_url+'/suggest/concepts'
    params = {
        'text': text
    }

    headers = {
        'Accept': 'application/json'
    }

    response = requests.get(url, params=params, headers=headers)

    # Check the response status code
    if response.status_code == requests.codes.ok:
        # Print the response content
        data = response.json()

        if len(data['suggestions']) > 0:
            return data['suggestions'][0]["url"]
        else:
            # Handle the case when the suggestions list is empty
            return None
    else:
        # Print an error message if the request was not successful
        return None


def get_sali_suggestions_for_concept(suggest_url,text):
    suggested_labels = []
    url = base_url+suggest_url
    params = {
        'text': text
    }

    headers = {
        'Accept': 'application/json'
    }

    response = requests.get(url, params=params, headers=headers)

    # Check the response status code
    if response.status_code == requests.codes.ok:
        # Print the response content
        data = response.json()
        suggestions = data['suggestions']
        for suggestion in suggestions:
            suggested_labels.append(suggestion["label"])    
        return suggested_labels
    else:
        # Print an error message if the request was not successful
        return []
    
def get_sali_suggestions(text):
    concept_url = get_sali_concept_url(text)
    if concept_url is not None:
        return get_sali_suggestions_for_concept(concept_url,text)
    else:
        return []
    
def get_list_of_ancestors(graph, tag_name):
    ancestor_list = []
    
    parent = get_parent_tag(graph,tag_name)
    while parent is not None:
        ancestor_list.append(parent)
        parent = get_parent_tag(graph,parent)
    return ancestor_list
        


def get_parent_tag(graph,tag_name):
    
    query_string = '''
        SELECT ?subject ?object ?predicate 
        WHERE {
            ?subject rdfs:label "%s".
            ?subject rdfs:subClassOf ?object.
            ?object rdfs:label ?predicate.   
        }
        ''' % tag_name

    # Execute the query and iterate over the results
    results = graph.query(query_string)

    if len(results) == 0:
        return None
    else:
        for result in results:
            return str(result.predicate) 

def get_child_tag(graph,tag_name):
    children_tags = []
    query_string = '''
        SELECT ?subject ?object ?predicate 
        WHERE {
            ?subject rdfs:label "%s".
            ?object rdfs:subClassOf ?subject.
            ?object rdfs:label ?predicate.   
        }
        ''' % tag_name

    # Execute the query and iterate over the results
    results = graph.query(query_string)

    if len(results) == 0:
        return children_tags
    else:
        for result in results:
            children_tags.append(str(result.predicate)) 
        return children_tags