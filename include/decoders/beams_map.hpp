
#ifndef _ASR_REALTIME_BEAMS_MAP
#define _ASR_REALTIME_BEAMS_MAP

#include <unordered_map>
#include <vector>
#include <algorithm>
#include <optional>
#include <set>
#include "decoders/beam.hpp"
#include "utils/my_utils.hpp"
#include "utils/fst_glog_safe_log.hpp"



using namespace asr;

namespace beam{


class BeamPtrMap{
/*
A class to store unique beams. When a new beam is added
the class checks if the beam's prefix exists or not.
If it exists, the p_b, and p_nb of the new beam is added
to the existing and then it is deleted. 
*/
private:
    typedef std::unordered_map<std::string, ctcBeam*>::iterator iterator;
    std::unordered_map<std::string, ctcBeam*> _beams_map;
    std::set<ctcBeam*> _beams_to_delete;


public:
    void add_beam(ctcBeam* beam){
        VLOG(5) << "[BeamPtrMap/add_beam]: adding " << beam->get_sequence() << " with address "
                << beam << " to the map";
        // if the element exist 
        auto it = _beams_map.find(beam->get_sequence());
        if (it != _beams_map.end()){
            VLOG(5) << "[BeamPtrMap/add_beam]: similar sequence exists " << " at " << it->second;

            // if new beam and existing beam point to the same memory 
            if (it->second == beam){
                VLOG(5) << "[BeamPtrMap/add_beam]: existing beam and new beam " 
                        << "point to the same meory address";
            }
            else{ // if new beam and existing beam hold different memories, add the new to garbage collector
                _beams_to_delete.insert(beam);
                VLOG(5) << "added the new beam at " << beam << " to the garbage collector.";
            }
        }
        else{
            VLOG(5) << "[BeamPtrMap/add_beam]: no similar beam exists. adding new beam directly";
            _beams_map[beam->get_sequence()] = beam;

        }
       return;
    };

    void clear(){_beams_map.clear();}

    ctcBeam* find_beam(std::string sequence){
        auto it = _beams_map.find(sequence);
        if (it != _beams_map.end()){ // if a beam stroing the same sequence exists 
            return it->second; // return a ptr to the beam stroing the sequence 
        }
        else{
            return nullptr;
        }
    }

    void clean_garbage(){
        for (auto& beam_to_delete : _beams_to_delete){
            VLOG(5) << "[BeamPtrMap/clean_garbage]: " 
                    << "deleting beam " << beam_to_delete->get_sequence()
                    << " at " << beam_to_delete;
            delete beam_to_delete;
        }
        _beams_to_delete.clear(); // important as the delted memory might (and probably will) br used by a new variable
    }

    iterator begin(){return _beams_map.begin();}
    
    iterator end(){return _beams_map.end();}

};





class BeamsMapWrapper{

    typedef std::unordered_map<std::string, beam::ctcBeam> BeamsMap;
private:
    BeamsMap _beams_map;
    int _beams_width;
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
            if (beam.get_score() > max_val.value()) LOG(WARNING) << "val is greater than max val";
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
            VLOG(6) << "[BeamsMapWrapper/update_beams_map]: adding its score (" << beam.get_score() << ") to the score vector";
            _beams_score.push_back(beam.get_score());
        }
        else{
            _beams_map[beam.get_sequence<std::string>()] += beam; // FIXME: what happens to the other
            VLOG(6) << "[BeamsMapWrapper/update_beams_map]: adding its score (" << beam.get_score() << ") to the score vector";
            _beams_score.push_back(_beams_map[beam.get_sequence<std::string>()].get_score());
        }

        // update the beams score vector 
        
    }

private:
    void clear_scores(){
        _beams_score.clear();
    }

};


}


#endif // _ASR_REALTIME_BEAMS_MAP