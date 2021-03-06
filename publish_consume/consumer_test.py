import pandas as pd
from pandas.compat import StringIO
from kafka import KafkaConsumer
from utils.data import gen_test_entry

def main():
    print("==> Spoofing detection on!")

    topic_name = 'raw_nmea'

    # if we want the consumer to run forever remove consumer_timeout_ms
    consumer = KafkaConsumer(topic_name, auto_offset_reset='earliest', bootstrap_servers=['localhost:9092'],
                             consumer_timeout_ms=1000)

    for msg in consumer:
        entry = msg.value.decode('utf-8')

        data_test = gen_test_entry(entry)
        if data_test == "":
            continue

        X_test = pd.read_csv(StringIO(data_test))

        print(X_test)
        break

    consumer.close()

    return


if __name__ == '__main__':
    main()
