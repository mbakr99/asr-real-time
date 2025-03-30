#ifndef _ASR_REALTIME_BEAMS_MAP
#define _ASR_REALTIME_BEAMS_MAP

#include <unordered_map>
#include <vector>
#include <algorithm>
#include <optional>
#include "decoders/beam.hpp"
#include "utils/my_utils.hpp"
#include "utils/fst_glog_safe_log.hpp"




namespace beam{


class BeamsMapWrapper{

    typedef std::unordered_map<std::string, beam::ctcBeam> BeamsMap;
private:
     BeamsMap _beams_map;
    int _beams_width = 0;
    std::vector<double> _beams_score;


public:
    // constructors 
    BeamsMapWrapper() : _beams_width(10) {
        LOG(INFO) << "[BeamsMapWrapper/constructor]: (default) instance has been created with " << _beams_width << " beams";
    }; 
    BeamsMapWrapper(int num_beams) : _beams_width(num_beams){
        LOG(INFO) << "[BeamsMapWrapper/constructor]: instance has been created with " << _beams_width << " beams";
    }
    ~BeamsMapWrapper(){};

    // setters 
    void set_beams_width(int beams_width){
        _beams_width = beams_width;
    }

    // getters
    std::optional<double> get_min(){
        if (_beams_score.empty()){
            LOG(WARNING) << "[BeamsMapWrapper/get_min]: scores have not been update. There might not be beams";
            return std::nullopt;
        }
        else{
            auto min_result =  std::min_element(_beams_score.begin(), _beams_score.end());
            return *min_result;
        }
    }
    std::optional<double> get_max(){
        if (_beams_score.empty()){
            LOG(WARNING) << "[BeamsMapWrapper/get_min]: scores have not been update. There might not be beams";
            return std::nullopt;
        }
        else{
            auto max_result =  std::max_element(_beams_score.begin(), _beams_score.end());
            return *max_result;
        }
    }
    BeamsMap get_map(){
        return _beams_map;
    }
    size_t size(){
        return _beams_map.size();
    }
    bool is_empty(){
        return _beams_map.empty();
    }
    
    // functionality 
    void scale_beams_score(const int& lower_bound, const int& upper_bound){
        // map the beams score to a desired rnage
        auto min_val = get_min();
        auto max_val = get_max();
        if (!min_val.has_value()) return;
        VLOG(5) << "[BeamsMapWrapper/scale_beams_score]: scaling the score to fall in [" 
                << lower_bound << ", " << upper_bound << "]." << "\n"
                << "min val: " << min_val.value() << ", max_val: " << max_val.value();
        for (auto& [sequence, beam] : _beams_map){
            beam.set_score(myutils::map_to_range(beam.get_score(), 
                           min_val.value(), max_val.value(), 
                           lower_bound, upper_bound)); 
        }

        // clear the _beams_score vector 
        clear_scores();
    }
    void clear_beams_map(){
        VLOG(6) << "[BeamsMapWrapper/clear_beams_map]: clearing beams map";
        if (!_beams_map.empty()) _beams_map.clear();
        if (!_beams_score.empty()) clear_scores();
    }
    void update_beams_map(const beam::ctcBeam& beam){
        VLOG(6) << "[BeamsMapWrapper/update_beams_map]: adding " << beam.get_sequence() << " to the beams map";
        // insert the beam
        if (_beams_map.find(beam.get_sequence<std::string>()) == _beams_map.end()){ // key does not exist
            _beams_map.insert({beam.get_sequence<std::string>(), beam});
        }
        else{
            _beams_map[beam.get_sequence<std::string>()] += beam; // FIXME: what happens to the other
        }

        // update the beams score vector 
        VLOG(6) << "[BeamsMapWrapper/update_beams_map]: adding its score (" << beam.get_score() << ") to the beams map";
        _beams_score.push_back(beam.get_score());
    }

private:
    void clear_scores(){
        _beams_score.clear();
    }

};


}


#endif // _ASR_REALTIME_BEAMS_MAP