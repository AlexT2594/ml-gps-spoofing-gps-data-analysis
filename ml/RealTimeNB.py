from utils.data import nmea_log_to_entry
import multiprocessing as mp
from threading import Thread
from kafka import KafkaConsumer
from utils.ml import BasicCountVectorizer
from sklearn.naive_bayes import MultinomialNB

topic_name = 'raw_nmea_numeric'


def main(data_train, y_train):

    vectorizer = BasicCountVectorizer()
    vectorizer.fit(data_train)
    X_train = vectorizer.transform(data_train)

    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    queue = mp.Queue()

    thread = Thread(target=consumeData, args=(queue,))
    thread.start()

    while True:
        try:
            test_entry = queue.get(timeout=10)
        except:
            print("    Timeout expired, closing application.")
            break

        X_test = vectorizer.transform(test_entry)
        pred = clf.predict(X_test)

        if pred[0] == 0:
            pred = "True"
        else:
            pred = "Spoofed"

        print("    Prediction: " + pred)

    print("    Exiting...")
    return


def consumeData(queue):

    consumer = KafkaConsumer(topic_name, auto_offset_reset='earliest', bootstrap_servers=['localhost:9092'],
                             consumer_timeout_ms=10000)

    for msg in consumer:
        entry = msg.value.decode('utf-8')

        entry = nmea_log_to_entry(entry)

        if len(entry) == 0:
            continue

        test_entry = [entry['$GPGGA'] + entry['$GPGSV'] + entry['$GPGSA'] + entry['$GPRMC']]

        queue.put(test_entry)

    consumer.close()


if __name__ == '__main__':
    main()
