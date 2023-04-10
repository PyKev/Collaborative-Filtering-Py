import pandas as pd
from scipy.spatial import distance
from scipy import spatial

df = pd.read_csv('ratings_small.csv')
df = df.drop('timestamp', axis=1)
print(df,'\n')
#dataframe de peliculas vistas
watched_movies = pd.pivot_table(df,index='userId',columns='movieId',values='rating',aggfunc='sum')
watched_movies = watched_movies.fillna(0)
print(watched_movies,'\n')
#Lista de peliculas que las han visto mas de 50 personas
mask = watched_movies.astype(bool).sum(axis=0).ge(50)
movies_rated_by_50_users = watched_movies.loc[:, mask].columns.tolist()

#Lista de peliculas que las han visto mas de 300 personas
mask = watched_movies.astype(bool).sum(axis=0).ge(300)
movies_rated_by_300_users = watched_movies.loc[:, mask].columns.tolist()

#Lista de peliculas no vistas por un usuario n
def not_watched(user):
    user_row = watched_movies.loc[user]
    false_columns = user_row.index[user_row == 0].tolist()
    return false_columns
#Lista de peliculas vistas por un usuario n
def watched(user):
    user_row = watched_movies.loc[user]
    true_columns = user_row.index[user_row != 0].tolist()
    return true_columns

#Selecciona al usuario de id 1
not_watched_user1 = not_watched(1)
watched_user1 = watched(1)

#Peliculas que el usuario 1 no ha visto pero que si la han visto más de 300 usuarios
not_movies_check = [not_watched for not_watched in not_watched_user1 if not_watched in movies_rated_by_300_users]
print("No vistas por usuario 1, pero vistas por más de 300 usuarios: \n", not_movies_check)
#Peliculas que el usuario 1 ha visto y que tambien la han viso mas de 50 personas
movies_check = [watched for watched in watched_user1 if watched in movies_rated_by_50_users]
print("Vistas por usuario 1 y por más de 50 usuarios: \n", movies_check,'\n')

#Encontrar los usuarios que han visto las pelis 1339, 2294, 3671, 356 y 318
print("Usuarios que vieron las peliculas con ID 1339, 2294, 3671, 356 y 318:",'\n')
newdf = watched_movies.loc[(watched_movies[[1339, 2294, 3671, 356,318]] != 0).all(axis=1)][[1339, 2294, 3671, 356,318]]
print(newdf,'\n')
#Los usuarios escogidos fueron 15,73,130 y 195

users_to_select = [15,73,130,195,468,472,580,1]
columns_to_select = [1339, 2294, 3671,356,318]
dataset = watched_movies.loc[users_to_select,columns_to_select]
print(dataset,'\n')

arrays_watched = dataset.values[:,:3]
print(arrays_watched,'\n')
#dataset["Distances"] = [distance.euclidean(arrays[4], arrays[a]) for a in range(len(arrays))]
dataset["Sim Coseno"] = [1-spatial.distance.cosine(arrays_watched[7],arrays_watched[a]) for a in range(len(arrays_watched))]
dataset["M1"] = dataset[356]*dataset["Sim Coseno"]
dataset["M2"] = dataset[318]*dataset["Sim Coseno"]
suma1 = dataset["M1"].sum()
suma2 = dataset["M2"].sum()
print(dataset,'\n')
sum_users = dataset["Sim Coseno"].sum()-1
puntaje1 = suma1/sum_users
puntaje2 = suma2/sum_users
print("Puntaje pelicula 356:", puntaje1)
print("Puntaje pelicula 318:", puntaje2,'\n')
print("Se recomendaría la pelicula 318 pues tendría mayor aceptación por el usuario 1")


