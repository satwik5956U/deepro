from django.contrib import admin
from django.urls import path,include
from . import views
urlpatterns = [
    path('get/',views.getBlogs),
    path('post/',views.postBlogs),
    path('put/<int:pk>/',views.putBlog),
    path('delete/<int:pk>/',views.deleteBlog),
]
