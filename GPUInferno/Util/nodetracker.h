#pragma once

#include <iostream>
#include <set>
#include <string>
#include <map>

namespace Inferno {
    class NodeTracker {
    private:
        // Changed from a set to a map to associate IDs with names
        static std::map<int, std::string> idNameMap;

    public:
        NodeTracker();

        // Original method for backward compatibility
        static void addID(int id);

        // New method that takes both ID and name
        static void addID(int id, const std::string& name);

        // Remove an ID (and its associated name)
        static void removeID(int id);

        // Update the name for an existing ID
        static void updateName(int id, const std::string& name);

        // Get the name associated with an ID
        static std::string getName(int id);

        // Check if an ID exists
        static bool hasID(int id);

        // Display all IDs and their associated names
        static void dumpIDs();
    };
}

