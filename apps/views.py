from django.shortcuts import render
from apps.classification import main

# def form_index(request):
#     return render(request, 'apps/classification.html')

def classification(request):
    if request.method == 'POST':
        synopsis = request.POST['synopsis']
        genre = main.main(synopsis)
        content = {'genre': genre}
        return render(request, 'apps/classification.html', content)
    return render(request, 'apps/classification.html')

def index(request):
    return render(request, 'apps/index.html')

