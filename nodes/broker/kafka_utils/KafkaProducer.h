#pragma once

#ifdef VP_WITH_KAFKA
#include <string>
#include <iostream>

// compile tips:
// run `apt-get install librdkafka-dev`, v0.11.3 for ubuntu 18.04 by default.
#include <librdkafka/rdkafkacpp.h>

// wrapper class for Producer in librdkafka
class KafkaProducer
{
public:
	explicit KafkaProducer(const std::string& brokers, const std::string& topic, int partition);

	void pushMessage(const std::string& str);
	~KafkaProducer();


private:
	std::string m_brokers;
	std::string m_topicStr;
	int m_partition;

	RdKafka::Conf* m_config;
	RdKafka::Conf* m_topicConfig;
	RdKafka::Topic* m_topic;
	RdKafka::Producer* m_producer;

	RdKafka::DeliveryReportCb* m_dr_cb;
	RdKafka::EventCb* m_event_cb;
	RdKafka::PartitionerCb* m_partitioner_cb;
};
#endif