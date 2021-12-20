from rest_framework import serializers


class BotSerializer(serializers.Serializer):
    text = serializers.CharField(required=True)
    type = serializers.CharField(required=True)
    size = serializers.IntegerField(default = 1, required=False)
    