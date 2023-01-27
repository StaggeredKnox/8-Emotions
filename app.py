import os
import random
import librosa
import tensorflow as tf
from flask import Flask, render_template, redirect, request
from numpy import newaxis, argmax

app = Flask(__name__)


def predict(audioFileName):
    model = tf.keras.models.load_model("EMOmodel.h5", compile=False)
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0001),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    input_data, input_sr = librosa.load(audioFileName)

    codes = ["angry", "sad", "happy", "surprised", "neutral", "disgust", "fear", "calm"]
    sec_2 = 2 * input_sr

    n = int(len(input_data) / sec_2)
    T = 0
    ans = {
        "start": [],
        "end": [],
        "emotion": []
    }

    for i in range(n):
        signal = input_data[i * sec_2: (i + 1) * sec_2]

        input_mfccs = librosa.feature.mfcc(signal, input_sr, n_mfcc=45, n_fft=2048, hop_length=512)
        input_mfccs = input_mfccs[newaxis, ..., newaxis]
        input_mfccs = input_mfccs.T

        j = argmax(model.predict(input_mfccs)[0])

        T += 2
        mn, sc = divmod(T, 60)
        hr, mn = divmod(mn, 60)
        t = f"{hr}:{mn}:{sc}"

        if len(ans["start"]) == 0:
            ans["start"].append("0:0:0")
            ans["end"].append("0:0:2")
            ans["emotion"].append(codes[j])
        else:
            if ans["emotion"][-1] == codes[j]:
                ans["end"][-1] = t
            else:
                ans["start"].append(ans["end"][-1])
                ans["end"].append(t)
                ans["emotion"].append(codes[j])

    os.remove(audioFileName)

    if len(ans) == 0:
        ans["start"].append("0")
        ans["end"].append("0")
        ans["emotion"].append("null")

    return ans


@app.route("/", methods=['GET', 'POST'])
def index():
    answer= {"start": [],
             "end": [],
             "emotion": []
             }
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        audioFile = request.files["file"]
        if audioFile.filename == "":
            return redirect(request.url)

        if audioFile:
            audioFileName = str(random.randint(0, 1000))
            audioFile.save(audioFileName)
            answer = predict(audioFileName)

    return render_template("index.html", answer=answer)


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
