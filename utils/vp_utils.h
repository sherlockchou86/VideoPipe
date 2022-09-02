#pragma once

#include <iostream>
#include <memory>
#include <opencv2/imgproc.hpp>
using namespace std;

namespace vp_utils {
    // string format in C++17
    template<typename ... Args>
    inline string string_format(const string& format, Args ... args){
        size_t size = 1 + snprintf(nullptr, 0, format.c_str(), args ...); 
        // unique_ptr<char[]> buf(new char[size]);
        char bytes[size];
        snprintf(bytes, size, format.c_str(), args ...);
        return string(bytes);
    }

    // get optimal font scale depend on screen width and height
    inline double get_optimal_font_scale(string text, int screen_width, int screen_height, cv::Size & text_size, int fontface = cv::FONT_HERSHEY_PLAIN, int thickness = 1) {
        int baseline = 0;
        for (int i = 20; i > 0; i--) {
            auto size = cv::getTextSize(text, fontface, i / 20.0, thickness, &baseline);
            if (size.width < screen_width && size.height < screen_height) {
                text_size = size;
                return i / 20.0;
            }
        }
        return 1;
    }

    // draw rounded rectangle, fill the backgound.
    inline void draw_rounded_rectangle(cv::Mat& src, cv::Point top_left, cv::Point bottom_right, int corner_radius, cv::Scalar color = cv::Scalar(), int thickness = 1, cv::Scalar fill_color = cv::Scalar(255, 255, 255), int line_type = cv::LINE_4)
    {
         /* corners:
          * p1 - p2
          * |     |
          * p4 - p3
          */
        if (corner_radius < 1) {
            corner_radius = 1;
        }
        
        cv::Point p1 = top_left;
        cv::Point p2 = cv::Point (bottom_right.x, top_left.y);
        cv::Point p3 = bottom_right;
        cv::Point p4 = cv::Point (top_left.x, bottom_right.y);

        // fill rounded rectangle 
        if (thickness < 0) {
            cv::rectangle(src, cv::Rect(cv::Point (p1.x + corner_radius, p1.y), cv::Point (p3.x - corner_radius, p3.y)), fill_color, thickness);
            cv::rectangle(src, cv::Rect(cv::Point (p1.x, p1.y + corner_radius), cv::Point (p3.x, p3.y - corner_radius)), fill_color, thickness);
            cv::ellipse( src, p1 + cv::Point(corner_radius, corner_radius), cv::Size( corner_radius, corner_radius ), 180.0, 0, 90, fill_color, thickness, line_type );
            cv::ellipse( src, p2 + cv::Point(-corner_radius, corner_radius), cv::Size( corner_radius, corner_radius ), 270.0, 0, 90, fill_color, thickness, line_type );
            cv::ellipse( src, p3 + cv::Point(-corner_radius, -corner_radius), cv::Size( corner_radius, corner_radius ), 0.0, 0, 90, fill_color, thickness, line_type );
            cv::ellipse( src, p4 + cv::Point(corner_radius, -corner_radius), cv::Size( corner_radius, corner_radius ), 90.0, 0, 90, fill_color, thickness, line_type );

            thickness = 1;
        }
 
        // draw straight lines
        cv::line(src, cv::Point (p1.x + corner_radius, p1.y), cv::Point (p2.x - corner_radius, p2.y), color, thickness, line_type);
        cv::line(src, cv::Point (p2.x, p2.y + corner_radius), cv::Point (p3.x, p3.y - corner_radius), color, thickness, line_type);
        cv::line(src, cv::Point (p4.x + corner_radius, p4.y), cv::Point (p3.x - corner_radius, p3.y), color, thickness, line_type);
        cv::line(src, cv::Point (p1.x, p1.y + corner_radius), cv::Point (p4.x, p4.y - corner_radius), color, thickness, line_type);

        // draw arcs
        cv::ellipse( src, p1 + cv::Point(corner_radius, corner_radius), cv::Size( corner_radius, corner_radius ), 180.0, 0, 90, color, thickness, line_type );
        cv::ellipse( src, p2 + cv::Point(-corner_radius, corner_radius), cv::Size( corner_radius, corner_radius ), 270.0, 0, 90, color, thickness, line_type );
        cv::ellipse( src, p3 + cv::Point(-corner_radius, -corner_radius), cv::Size( corner_radius, corner_radius ), 0.0, 0, 90, color, thickness, line_type );
        cv::ellipse( src, p4 + cv::Point(corner_radius, -corner_radius), cv::Size( corner_radius, corner_radius ), 90.0, 0, 90, color, thickness, line_type );
    }

    // put text at the center of specific rect on canvas
    inline void put_text_at_center_of_rect(cv::Mat& canvas, string text, cv::Rect rect, bool fill = false, int font_face = cv::FONT_HERSHEY_PLAIN, int thickness = 1, cv::Scalar color = cv::Scalar(), cv::Scalar border_color = cv::Scalar(201, 201, 205), cv::Scalar fill_color = cv::Scalar(201, 201, 205), int padding = 1) {
        cv::Size text_size;
        auto font_scale = get_optimal_font_scale(text, rect.width - padding * 2, rect.height - padding * 2, text_size, font_face, thickness);
        if (fill) {
            draw_rounded_rectangle(canvas, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), 3, border_color, -1, fill_color);
        }
        cv::putText(canvas, 
            text, 
            cv::Point(rect.x + (rect.width - text_size.width) / 2, rect.y + rect.height - (rect.height - text_size.height) / 2), 
            font_face, font_scale, color);
    }

    inline std::string round_any(double input, int precision) {
        input = input * std::pow(10, precision) + 0.5;
        auto output = std::to_string(std::floor(input) / std::pow(10, precision));

        output.erase(std::find_if(output.rbegin(), output.rend(), [](unsigned char ch) {
            return ch != '0';
        }).base(), output.end());

        return output;
    }
}