import os
print("Current working directory:", os.getcwd())

from hybrid_cf import recommend_items_hybrid #as get_recommendations
from hybrid_cf import (
    ratings_matrix,
    user_id_to_index,
    item_id_to_index,
    user_similarity_matrix,
    item_similarity_matrix,
    movie_id_to_title,
    index_to_movie_id
)

from tmdb_utils import get_movie_poster
from poster_fetcher import get_poster_url

from flask import Flask, request, render_template, url_for
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')
    
#def home():
 #   return "Hello, this is your movie recommender!"


@app.route('/search')
def search():
   return render_template('search.html')

@app.route('/search_results', methods=['GET'])
def search_results():
    title = request.args.get('title', '').strip().lower()
    if not title:
        return render_template('error.html', message="No title provided")

    matched_title = None
    for movie in movie_id_to_title.values():
        if title in movie.lower():
            matched_title = movie
            break

    if matched_title:
        poster_url = get_movie_poster(matched_title)
        if not poster_url:
            poster_url = url_for('static', filename='no_poster.png')
        return render_template('movie_details.html', title=matched_title, poster=poster_url)
    else:
        return render_template('error.html', message="No matching movie found")



@app.route('/recommend', methods=['GET'])
def recommend():
     user_id = request.args.get('user_id')
     if not user_id:
         return render_template('error.html', message="user_id is not provided")
    
     try:
        user_id_int = int(user_id)
        recommendations = recommend_items_hybrid(
            user_id=user_id_int,
            ratings_matrix=ratings_matrix,
            user_id_to_index=user_id_to_index,
            item_id_to_index=item_id_to_index,
            user_similarity_matrix=user_similarity_matrix,
            item_similarity_matrix=item_similarity_matrix,
            movie_id_to_title=movie_id_to_title,
            index_to_movie_id=index_to_movie_id,
            alpha=0.5,
            n_recommendations=5
        )
        print("Raw recommendations:", recommendations)


        rec_with_posters = []
        for title, rating in recommendations:
            poster_url = get_movie_poster(title)
            #poster_url = get_poster_url(title)
            if not poster_url:
                poster_url = url_for('static', filename='no_poster.png')         #fallback
            print(f"Title: {title}, Poster URL: {poster_url}")           #check if poster is broken
            rec_with_posters.append({"title": title, "rating": rating, "poster": poster_url})
        return render_template('recommendations.html', user_id=user_id, recommendations=rec_with_posters)
        
        #recommendations = get_recommendations(user_id) 
        #return render_template('recommendations.html', user_id = user_id, recommendations = rec_with_posters)
        
     except Exception as e:
        return render_template('error.html', message=str(e))


if __name__ == '__main__':
    app.run(debug=True)
