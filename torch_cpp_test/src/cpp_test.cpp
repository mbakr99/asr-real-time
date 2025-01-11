#include <iostream>
#include <queue>
#include <vector>
#include <algorithm>


int main(){

    std::vector<std::pair<float,std::string>> replace;

    std::unordered_map<std::string,float> dict{
        {"abc",2.5}, {"ab-",2.3}, {"abb",4.0}
    };

    for (auto& beam : dict){
        std::cout << beam.first << ": " << beam.second << std::endl;
    }


    std::priority_queue<
                        std::pair<float,std::string>,
                        std::vector<std::pair<float,std::string>>,
                        std::greater<>
                        > min_heap;
    for (auto& [seq,score] : dict){
        min_heap.emplace(score,seq);
        if (min_heap.size() > 2)
            min_heap.pop();
    }

    std::vector<std::pair<std::string,float>> top_k;
    while(!min_heap.empty()){
        top_k.emplace_back(min_heap.top().second,min_heap.top().first);
        min_heap.pop();
    }

    for (auto it = top_k.rbegin(); it != top_k.rend(); it++){
        std::cout << it->first << std::endl;
    }
    

}
