json.posts @posts do |post|
  json.id post.unique_id
  json.creator post.creator
  json.verified post.verified
  json.followers post.followers
  json.followers post.following
  json.impressions post.impressions
  json.upvotes post.upvotes
  json.reposts post.reposts
  json.creator_score post.creator_score
  json.reach_score post.reach_score
  json.body post.body
  json.hashtags post.hashtags
end