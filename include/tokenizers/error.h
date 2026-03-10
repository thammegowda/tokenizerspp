#pragma once
/// @file tokenizers/error.h
/// Error type for the tokenizerpp library.

#include <source_location>
#include <string>
#include <expected>

namespace tokenizers {

/// Error type used throughout the library.
class Error {
public:
    explicit Error(std::string message,
                   std::source_location loc = std::source_location::current())
        : message_(std::move(message)), location_(loc) {}

    Error(const Error&) = default;
    Error(Error&&) noexcept = default;
    Error& operator=(const Error&) = default;
    Error& operator=(Error&&) noexcept = default;

    [[nodiscard]] const std::string& message() const noexcept { return message_; }
    [[nodiscard]] const std::source_location& location() const noexcept { return location_; }

    [[nodiscard]] std::string to_string() const {
        return std::string(location_.file_name()) + ":" +
               std::to_string(location_.line()) + ": " + message_;
    }

private:
    std::string message_;
    std::source_location location_;
};

/// Convenience: create an std::unexpected<Error>.
inline std::unexpected<Error> make_error(
    std::string message,
    std::source_location loc = std::source_location::current()) {
    return std::unexpected<Error>(Error(std::move(message), loc));
}

} // namespace tokenizers
