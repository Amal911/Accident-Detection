from django.shortcuts import render

from .utils import startapplication


# Create your views here.
def home(request):
    return render(request,'home.html')

def detect(request):
    startapplication()
    return render(request,'home.html')