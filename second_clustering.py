# import pandas as pd
# from sklearn.cluster import KMeans
# import numpy as np
# import ast
#
# def cluster_paragraphs(df_encoded_output):
#     kmeans = KMeans(n_clusters=6, random_state=0, n_init="auto")
#     kmeans.fit(np.array(df_encoded_output["para_embedding"].tolist()))
#
#     df_encoded_output["second_clustering_labels"] = kmeans.labels_
#
#     return df_encoded_output
#
#
# df = pd.DataFrame(pd.read_csv('/Users/shashankshandilya/PycharmProjects/Information-Extraction/Output_vector_files/second_cluster_input.csv'))
#
# res = df['para_embedding'].tolist()
# opt = []
#
# for string in res:
#     opt.append(ast.literal_eval(string))
#
# # print(df['para_embedding'])
# # print(opt)
# df['para_embedding'] = opt
#
# rslt_df = df[df['k_means_labels'] == 2][['para_text', 'para_embedding', 'k_means_labels']]
# print(rslt_df)
#
# df_clustered_output = cluster_paragraphs(rslt_df)
# df_clustered_output[["para_text", "para_embedding", "k_means_labels", "second_clustering_labels"]].to_csv("/Users/shashankshandilya/PycharmProjects/Information-Extraction/Output_vector_files/hierachical_cluster_output.csv")

