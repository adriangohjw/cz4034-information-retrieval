module SearchHelper

  def self.empty_params(search_term:, hashtags:)
    return false unless search_term.blank?
    return false unless hashtags.count == 0
    return true
  end

  def self.concat_hash_into_array(*hashes)
    array_result = Array.new

    hashes.each do |hash|
      array_result << hash if !hash.blank?
    end
    
    return array_result
  end

  def self.querystring_to_hash(querystring)
    return nil if querystring.blank?
    return {
      query_string: {
        query: "#{querystring}"
      }
    }
  end

  class ArrayParam
    attr_reader :value

    def initialize(param_name, array_param)
      @param_name = param_name
      @array_param = array_param
      @value = parsed_array_param
    end

    def parsed_array_param
      results = Array.new
      @array_param.each do |param| 
        results << { 
          bool: {
            should: [
              {
                match_phrase: {
                  "#{@param_name}": param
                }
              }
            ]
          }
        }
      end
      return results
    end
  end

end