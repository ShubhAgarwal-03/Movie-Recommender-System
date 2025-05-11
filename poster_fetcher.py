import requests

OMDB_API_KEY = ""  # Replace with your real API key

def get_poster_url(title):
    url = f"http://www.omdbapi.com/?apikey={OMDB_API_KEY}&t={title}"
    try:
        response = requests.get(url)
        data = response.json()
        return data.get('Poster', 'https://via.placeholder.com/200x300?text=No+Image')
    except Exception:
        return 'https://via.placeholder.com/200x300?text=Error'
