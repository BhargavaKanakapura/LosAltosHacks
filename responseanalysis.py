import cohere

co = cohere.Client("qnwyPerImbDPtxYJQ2c0bix4LD10GR8cwugHU1xN")

from cohere.responses.classify import Example
import pandas as pd

sentiment_file_path = 'IMDB Dataset.csv'
sentiment_df = pd.read_csv(sentiment_file_path).sample(frac=1.0)
sentiment_examples = [
    Example(sentiment_df['review'][i], sentiment_df['sentiment'][i])
    for i in range(len(sentiment_df['sentiment']))
][:666]

catagories_file_path = "complaints_processed.csv"
catagories_df = pd.read_csv(catagories_file_path).sample(frac=1.0)
catagories_examples = [
    Example(catagories_df['narrative'][i], catagories_df['product'][i])
    for i in range(len(catagories_df['product']))
][:666]

emotions_file_path = "tweet_emotions.csv"
emotions_df = pd.read_csv(emotions_file_path).sample(frac=1.0)
emotions_examples = [
    Example(emotions_df['content'][i], emotions_df['sentiment'][i])
    for i in range(len(emotions_df['sentiment']))
][:666]

inputs = [
    "Rithvik Chavali scammed me; im really mad about it",
    "Darsh Gupta is being a pain in the ass; please help",
    "Darsh Gupta was very helpful in the afterlife",
    "Bhargava came waaaayyyy to close to me; I felt uncomfortable",
    "The product was broken when it arrived"
]


def get_scores(inputs, examples=sentiment_examples):

    response = co.classify(
        model='large',
        inputs=inputs,
        examples=examples,
    )

    return [
        (-1 if response.classifications[i].prediction == "negative" else 1) *
        (response.classifications[i].confidence**5) * 5 + 5
        for i in range(len(inputs))
    ]


def get_catagories(inputs, examples=catagories_examples):

    response = co.classify(
        model='large',
        inputs=inputs,
        examples=examples,
    )

    return [(response.classifications[i].prediction,
             response.classifications[i].confidence)
            for i in range(len(inputs))]


def get_emotions(inputs, examples=emotions_examples):

    response = co.classify(
        model='large',
        inputs=inputs,
        examples=examples,
    )

    return [(response.classifications[i].prediction,
             response.classifications[i].confidence)
            for i in range(len(inputs))]


def summarise(inputs):

    outputs = []
    for i in inputs:
        if (len(i) > 250):
            outputs.append(
                co.summarize(
                    text=i,
                    length='auto',
                    format='auto',
                    model='summarize-xlarge',
                    additional_command='',
                    temperature=0.3,
                ))
        else:
            outputs.append(i)
    return outputs

print(get_scores(inputs))
print(get_catagories(inputs))
print(get_emotions(inputs))
print(summarise(inputs))
