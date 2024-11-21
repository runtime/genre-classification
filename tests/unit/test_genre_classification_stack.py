import aws_cdk as core
import aws_cdk.assertions as assertions

from genre_classification.genre_classification_stack import GenreClassificationStack

# example tests. To run these tests, uncomment this file along with the example
# resource in genre_classification/genre_classification_stack.py
def test_sqs_queue_created():
    app = core.App()
    stack = GenreClassificationStack(app, "genre-classification")
    template = assertions.Template.from_stack(stack)

#     template.has_resource_properties("AWS::SQS::Queue", {
#         "VisibilityTimeout": 300
#     })
