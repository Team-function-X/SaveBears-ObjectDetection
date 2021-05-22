import uuid


class RekognitonClient(object):
    def __init__(self, boto3_client):
        self._boto3_client = boto3_client

    def get_image_labels(self, bucket, key):
        response = self._boto3_client.detect_labels(
            Image={
                'S3Object': {
                    'Bucket': bucket,
                    'Name': key
                },
            },
            MinConfidence=50.0
        )
        return [label['Name'] for label in response['Labels']]

    def get_image_emotions(self, bucket, key):
        response = self._boto3_client.detect_faces(
            Image={
                'S3Object': {
                    'Bucket': bucket,
                    'Name': key
                }
            },
            Attributes=[
                'ALL'
            ]
        )

        emotions = []

        for face in response['FaceDetails']:
            maxEmo = ''
            maxConfi = 0
            for emotion in face['Emotions']:
                if maxConfi < emotion['Confidence']:
                    maxEmo = emotion['Type']
                    maxConfi = emotion['Confidence']

            emotions.append(maxEmo)

        return emotions

    def start_video_label_job(self, bucket, key, topic_arn, role_arn):
        response = self._boto3_client.start_label_detection(
            Video={
                'S3Object': {
                    'Bucket': bucket,
                    'Name': key
                }
            },
            ClientRequestToken=str(uuid.uuid4()),
            NotificationChannel={
                'SNSTopicArn': topic_arn,
                'RoleArn': role_arn
            },
            MinConfidence=50.0
        )
        return response['JobId']

    def start_video_face_job(self, bucket, key, topic_arn, role_arn):
        response = self._boto3_client.start_face_detection(
            Video={
                'S3Object': {
                    'Bucket': bucket,
                    'Name': key
                }
            },
            ClientRequestToken=str(uuid.uuid4()),
            NotificationChannel={
                'SNSTopicArn': topic_arn,
                'RoleArn': role_arn
            },
            FaceAttributes='ALL'
        )
        return response['JobId']

    def get_video_job_labels(self, job_id):
        labels = set()
        client_kwargs = {
            'JobId': job_id,
        }
        response = self._boto3_client.get_label_detection(**client_kwargs)
        self._collect_video_labels(labels, response)
        while 'NextToken' in response:
            client_kwargs['NextToken'] = response['NextToken']
            response = self._boto3_client.get_label_detection(**client_kwargs)
            self._collect_video_labels(labels, response)
        return list(labels)

    def get_video_job_faces(self, job_id):
        labels = set()
        client_kwargs = {
            'JobId': job_id,
        }
        response = self._boto3_client.get_face_detection(**client_kwargs)
        self._collect_video_emotions(labels, response)
        while 'NextToken' in response:
            client_kwargs['NextToken'] = response['NextToken']
            response = self._boto3_client.get_face_detection(**client_kwargs)
            self._collect_video_emotions(labels, response)
        return list(labels)

    def _collect_video_labels(self, labels, response):
        for label in response['Labels']:
            label_name = label['Label']['Name']
            labels.add(label_name)

    def _collect_video_emotions(self, labels, response):
        for face in response['Faces']:
            emotions = face['Face']['Emotions']
            maxEmo = ''
            maxConfi = 0
            for emotion in emotions:
                if maxConfi < emotion['Confidence']:
                    maxEmo = emotion['Type']
                    maxConfi = emotion['Confidence']
            labels.add(maxEmo)