from django import forms

class UploadFileForm(forms.Form):
    video = forms.FileField()