import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def check_plagiarism(file):
    filename = file
    script_location = os.path.dirname(os.path.abspath('__file__'))
    test_file_location = f"{script_location}\\files\\{filename}"
    # test_file = file  #the original essay to be checked ... can be from any location
    test_text =open(test_file_location).read()

    path = f"{script_location}\\test_files_compare"    #the folder from where we compare the test essay to all essays
    # all_source = [path +'\\'+ str(doc) for doc in os.listdir(path) if doc.endswith('.txt','.pdf', '.doc')] # checks only with txt files
    all_source = [path +'\\'+ str(doc) for doc in os.listdir(path)]
    all_text =[open(File).read() for File in  all_source]

    final_source = [test_file_location]+all_source
    final_text = [test_text]+all_text

    vectorize = lambda Text: TfidfVectorizer().fit_transform(Text).toarray()
    similarity = lambda doc1, doc2: cosine_similarity([doc1, doc2])

    vectors = vectorize(final_text)
    zip_vectors = list(zip(final_source, vectors))
    plagiarism_results = set()
    source_a = zip_vectors[0][0]
    text_vector_a = zip_vectors[0][1]
    del zip_vectors[0]
    for source_b, text_vector_b in zip_vectors:
        if source_a==source_b:
            continue
        sim_score = similarity(text_vector_a, text_vector_b)[0][1]
        score = (source_a, source_b,sim_score)
        plagiarism_results.add(score)


    return plagiarism_results


 