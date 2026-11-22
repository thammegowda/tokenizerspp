#include <iostream>
#include <string>

#include <nlohmann/json.hpp>
#include "tokenizers/chat_template.h"

int main() {
    using json = nlohmann::ordered_json;
    std::string line;
    while (std::getline(std::cin, line)) {
        json response;
        try {
            const auto request = json::parse(line);
            tokenizers::ChatTemplate chat(request.at("template").get<std::string>(),
                                          request.value("bos_token", std::string{}),
                                          request.value("eos_token", std::string{}));
            auto result = chat.apply_json(request.at("messages"), request.value("add_generation_prompt", true),
                                          request.value("template_args", json::object()));
            if (result)
                response["rendered"] = *result;
            else
                response["error"] = result.error().message();
        } catch (const json::exception& error) {
            response["error"] = error.what();
        }
        std::cout << response.dump() << '\n';
    }
    if (!std::cin.eof()) {
        std::cerr << "Cannot read chat requests\n";
        return 1;
    }
}
