from kafka import KafkaProducer


def main():
    messages = get_nmea_entries("../data/lat_long_eval/lat_long_eval_G.txt")
    if len(messages) > 0:
        kafka_producer = connect_kafka_producer()

        for message in messages:
            publish_message(kafka_producer, 'raw_nmea_G', 'raw_nmea_message', message)

        if kafka_producer is not None:
            kafka_producer.close()


def publish_message(producer_instance, topic_name, key, value):
    try:
        key_bytes = bytes(key, encoding='utf-8')
        value_bytes = bytes(value, encoding='utf-8')
        producer_instance.send(topic_name, key=key_bytes, value=value_bytes)
        producer_instance.flush()
        #print("==> Message published successfully.")
    except Exception as ex:
        print("==> Exception in publishing message")
        print(str(ex))


def connect_kafka_producer():
    _producer = None

    try:
        _producer = KafkaProducer(bootstrap_servers=['localhost:9092'])
    except Exception as ex:
        print('Exception while connecting Kafka')
        print(str(ex))
    finally:
        return _producer


def get_nmea_entries(filename):
    entries = []

    file = open(filename, 'r')

    entry = ""

    for line in file.readlines():
        fields = line.split(',')
        message_ID = fields[0]

        line = line.rstrip()

        if message_ID == "$GPRMC":
            entry += line
            entries.append(entry)

            entry = ""
        else:
            entry += line + ";"

    print("Number of entries: " + str(len(entries)))

    return entries


if __name__ == '__main__':
    main()
