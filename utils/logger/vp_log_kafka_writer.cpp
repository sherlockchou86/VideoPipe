#ifdef VP_WITH_KAFKA
#include "vp_log_kafka_writer.h"

namespace vp_utils {
    
    vp_log_kafka_writer::vp_log_kafka_writer(/* args */) {

    }
    
    vp_log_kafka_writer::~vp_log_kafka_writer() {

    }

    void vp_log_kafka_writer::write(std::string log) {
        if (!inited) {
            throw "vp_log_kafka_writer not initialized!";
        }
        kafka_producer->pushMessage(log);
    }

    void vp_log_kafka_writer::init(std::string kafka_servers, std::string topic_name) {
        kafka_producer = std::make_shared<KafkaProducer>(kafka_servers, topic_name, 0);
        inited = true;
    }

    // for << operator
    vp_log_kafka_writer& vp_log_kafka_writer::operator<<(std::string log) {
        write(log);
        return *this;
    }
}
#endif