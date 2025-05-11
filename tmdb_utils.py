import requests

TMDB_API_KEY = "3b120d337645eb3be9e7d324a4648c6a"

def get_movie_poster(title):
    url = "https://api.themoviedb.org/3/search/movie"
    params = {
        "api_key": TMDB_API_KEY,
        "query": title
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json().get("results")
        if results:
            poster_path = results[0].get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception as e:
        print(f"Error fetching poster for {title}: {e}")
    return "/static/default_poster.jpg"  # fallback image
