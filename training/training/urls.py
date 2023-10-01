"""
URL configuration for training project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.http import HttpRequest
from django.urls import path
from ninja import NinjaAPI
from training.routes.datasets.default.columns import router as default_dataset_router
from training.routes.tabular.tabular import router as tabular_router

api = NinjaAPI()


@api.get("/test")
def test(request: HttpRequest):
    return 200, {"result": "200 Backend surface test successful"}


api.add_router("/datasets/default/", default_dataset_router)
api.add_router("/tabular", tabular_router)

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", api.urls),  # type: ignore
]
