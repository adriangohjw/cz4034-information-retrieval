class Post < ApplicationRecord

  searchkick

  SEARCH_RESULT_SIZE = 50

  def search_data
    {
      body: preprocessed_body,
      hashtags: hashtags,
      score: creator_score.to_f * reach_score.to_f,
      posted_at: posted_at
    }
  end

  def self.get_search_results(search_term:, hashtags: Array.new)

    if SearchHelper.empty_params(search_term: search_term, hashtags: hashtags)
      return Post.order(posted_at: :desc).first(SEARCH_RESULT_SIZE)
    end

    query = { 
      query: {
        function_score: {
          query: {
            bool: {
              must: SearchHelper.concat_hash_into_array(
                SearchHelper.querystring_to_hash(search_term)
              ) + SearchHelper::ArrayParam.new('hashtags', hashtags).value,
              filter: [],
              should: [],
              must_not: []
            }
          },
          field_value_factor: {
            field: "score",
            factor: 1.2,
            modifier: "square",
            missing: 1
          }
        }
      }
    }
    return self.search(body: query, per_page: SEARCH_RESULT_SIZE)
  end

  def preprocessed_body
    Post.remove_words_from_text(text: body, words: hashtags)
  end

  def self.update_all_scores
    followers_histogram = Post.followers_histogram[0]
    following_histogram = Post.following_histogram[0]
    impressions_histogram = Post.impressions_histogram[0]
    upvotes_histogram = Post.upvotes_histogram[0]
    reposts_histogram = Post.reposts_histogram[0]

    Post.all.each do |post|
      creator_score = post.calculate_creator_score(followers_histogram: followers_histogram,
                                                   following_histogram: following_histogram)
      reach_score = post.calculate_reach_score(impressions_histogram: impressions_histogram,
                                               upvotes_histogram: upvotes_histogram,
                                               reposts_histogram: reposts_histogram)
      post.update(creator_score: creator_score,
                  reach_score: reach_score)
    end
  end

  # score creator based on verified status, followers, following etc.
  def calculate_creator_score(followers_histogram: Post.followers_histogram[0], 
                              following_histogram: Post.following_histogram[0])
    normalized_followers_score(histogram_bins: followers_histogram) * 
    normalized_following_score(histogram_bins: following_histogram) *
    normalized_verified_score
  end

  # score post's impact and reach based on impressions, upvotes, reposts
  def calculate_reach_score(impressions_histogram: Post.impressions_histogram[0],
                            upvotes_histogram: Post.upvotes_histogram[0],
                            reposts_histogram: Post.reposts_array[0])
    normalized_impressions_score(histogram_bins: impressions_histogram) * 
    normalized_upvotes_score(histogram_bins: upvotes_histogram) * 
    normalized_reposts_score(histogram_bins: reposts_histogram)
  end

  def normalized_followers_score(histogram_bins: Post.followers_histogram[0])
    return 0 if histogram_bins.blank?
    Post.get_bin_position(value: self.followers, bins: histogram_bins).to_f / histogram_bins.size.to_f
  end

  def normalized_following_score(histogram_bins: Post.following_histogram[0])
    return 0 if histogram_bins.blank?
    (histogram_bins.size - Post.get_bin_position(value: self.following, bins: histogram_bins)).to_f / histogram_bins.size.to_f
  end

  def normalized_verified_score
    case self.verified
    when true
      return 1.0
    when false
      return 0.5
    else
      return 0.0
    end
  end

  def normalized_impressions_score(histogram_bins: Post.impressions_histogram[0])
    return 0 if histogram_bins.blank?
    (histogram_bins.size - Post.get_bin_position(value: self.impressions, bins: histogram_bins)).to_f / histogram_bins.size.to_f
  end

  def normalized_upvotes_score(histogram_bins: Post.upvotes_histogram[0])
    return 0 if histogram_bins.blank?
    Post.get_bin_position(value: self.upvotes, bins: histogram_bins).to_f / histogram_bins.size.to_f
  end

  def normalized_reposts_score(histogram_bins: Post.reposts_histogram[0])
    return 0 if histogram_bins.blank?
    Post.get_bin_position(value: self.reposts, bins: histogram_bins).to_f / histogram_bins.size.to_f
  end

  def self.followers_histogram(followers_array: Post.all.pluck(:followers), histogram_size: 100)
    Post.get_histogram(array: followers_array, size: histogram_size)
  end

  def self.following_histogram(following_array: Post.all.pluck(:following), histogram_size: 100)
    Post.get_histogram(array: following_array, size: histogram_size)
  end

  def self.impressions_histogram(impressions_array: Post.all.pluck(:impressions), histogram_size: 100)
    Post.get_histogram(array: impressions_array, size: histogram_size)
  end

  def self.upvotes_histogram(upvotes_array: Post.all.pluck(:upvotes), histogram_size: 100)
    Post.get_histogram(array: upvotes_array, size: histogram_size)
  end

  def self.reposts_histogram(reposts_array: Post.all.pluck(:reposts), histogram_size: 100)
    Post.get_histogram(array: reposts_array, size: histogram_size)
  end

  private

  def self.remove_words_from_text(text:, words:) 
    words&.each { |word| text.slice!(word) }
    return text;
  end

  def self.get_histogram(array:, size:)
    array.reject! { |x| x.nil? }
    (bins, freqs) = array.histogram(size)
  end

  def self.get_bin_position(value:, bins:)
    return nil if value.blank?
    
    counter = 0
    bins.each do |bin|
      break if value <= bin
      counter = counter + 1
    end
    return counter
  end

end
