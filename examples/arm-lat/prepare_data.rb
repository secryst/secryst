translit = File.readlines('./data/translit.la-hy.la')

translit = translit.join('').gsub("\n", '')


translit = translit.gsub('&br', '&')
translit = translit.gsub(/\&[a-z]*/, ' ')
translit = translit.downcase
translit = translit.gsub(/[0-9]{2,}/, '')
translit = translit.gsub(Regexp.new("[!@#$%^&*()_+=\\[{\\]};:<>|./?,\\-'\\\"]"), ' ')

translit = translit.split(' ')
File.write('./data/target.csv', translit.join("\n"))

arm = File.readlines('./data/translit.la-hy.hy')

arm = arm.join('').gsub("\n", '')

arm = arm.gsub('&br', '&')
arm = arm.gsub(/\&[a-z]*/, ' ')
arm = arm.downcase
arm = arm.gsub(/[0-9]{2,}/, '')
arm = arm.gsub(/՝(?!\p{L})/, ' ')
arm = arm.gsub(Regexp.new("[!@#$%^&*()_+=\\[{\\]};:<>|./?,\\-'\\\"։«»․]"), ' ')

arm = arm.split(' ')
File.write('./data/input.csv', arm.join("\n"))


inputs = File.readlines('./data/input.csv')
targets = File.readlines('./data/target.csv')
zip = inputs.zip(targets)
zip.delete_if do |el|
  ar = el[0]
  rom = el[1]
  if ar.match?(/^[0-9\–]+$/) ||
     ar.match?(/[a-z]/) ||
     ar.match?(/^[a-z0-9а-я]+$/) ||
     ar.match?(/^[–]+$/) ||
     ar.match?(/[‡×…ةт°\—№δก→ยเ•·璠وْعَ]/)
    true
  elsif rom.match?(/^[12][a-z]$/)
    true
  end
end

zip.map {|z| 
  z[0].gsub!(/[012356789`°²]+$/, '')
  z[1].gsub!(/[012356789`°²]+$/, '')
  # invisible characters down
  z[0].gsub!(/[​​‎]/, '') 
  z[1].gsub!(/[​​‎]/, '')

  if z[1].match?(/^[23][a-z]+$/)
    z[1] = z[1][1..-1]
    z[0] = z[0][1..-1]
  end
}

zip.uniq! {|z| z[0]}
File.write('./data/input.csv', zip.map {|z| z[0]}.join(''))
File.write('./data/target.csv', zip.map {|z| z[1]}.join(''))
